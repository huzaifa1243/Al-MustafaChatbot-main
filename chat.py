from functools import wraps
import os
from flask import session
import torch
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, abort, flash
import json
import random
from NeuralNet.model import NeuralNet
from NeuralNet.utils import tokenize, bag_of_words
import mimetypes


# File path to store user information
USERS_FILE = 'users.json'

def load_users():
    try:
        with open(USERS_FILE, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_users(users_data):
    with open(USERS_FILE, 'w') as file:
        json.dump(users_data, file)

# Load users from file
users = load_users()
# Define a dictionary of users (in a real application, you'd use a database)

app = Flask(__name__, static_url_path='/static')
# Set a secret key for the Flask application
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
# Define a dictionary to store admin users

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configurations
DATASET_PATH = os.getenv('DATASET_PATH', 'NeuralNet/dataset.json')
MODEL_PATH = os.getenv('MODEL_PATH', 'NeuralNet/data.pth')

from werkzeug.utils import secure_filename

# Define a folder to save uploaded files
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif','mp4','mp3'}



def load_data_and_model():
    try:
        with open(DATASET_PATH, 'r') as json_data:
            intents = json.load(json_data)
    except FileNotFoundError:
        logger.error("Error: dataset.json file not found.")
        intents = {"intents": []}  # Fallback or handle as necessary

    try:
        data = torch.load(MODEL_PATH)
    except FileNotFoundError:
        logger.error("Error: data.pth file not found.")
        data = None  # Fallback or handle as necessary

    if data:
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data['all_words']
        tags = data['tags']
        model_state = data["model_state"]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_state)
        model.eval()
    else:
        model, all_words, tags, device = None, [], [], torch.device('cpu')

    return intents, all_words, tags, model, device

def chatbot_response(sentence):
    intents, all_words, tags, model, device = load_data_and_model()

    if not model:
        return "Model not loaded. Please check the logs for errors."

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand your query. Please tell me in detail."

# Function to check if user is logged in
def login_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return decorated_function

def run_chatbot(question):
    response = chatbot_response(question)
    return response

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    question = request.json.get('question')
    if not question:
        abort(400, description="Invalid input")
    predicted_response = run_chatbot(question)
    return jsonify({'response': predicted_response})

@app.route('/admin')
@login_required
def admin():
    return render_template('admin.html')


@app.route('/create_admin_user', methods=['POST'])
@login_required
def create_admin_user():
    new_username = request.form.get('new_username')
    new_password = request.form.get('new_password')
    # Check if username already exists
    if new_username in users:
        print('Username already exists', 'error')
        return redirect(url_for('create_admin_user_page'))
    
    # Add new user to users dictionary
    users[new_username] = new_password
    # Save updated users to file
    save_users(users)
    print('New admin user created successfully', 'success')
    return redirect(url_for('admin'))




@app.route('/create_admin_user_page')
@login_required
def create_admin_user_page():
    return render_template('create_admin_user.html')

@app.route('/merit')
def merit():
    return render_template('merit.html')

@app.route('/update_response', methods=['POST'])
@login_required
def update_responses():
    intents, _, _, _, _ = load_data_and_model()
    updated_intents = intents.copy()

    for intent in updated_intents['intents']:
        tag_name = request.form.get(f'tag_{intent["tag"]}')
        if tag_name:
            intent['tag'] = tag_name

        updated_patterns = request.form.getlist(f'pattern_{intent["tag"]}[]')
        if updated_patterns:  # Check if updated patterns exist
            intent['patterns'] = updated_patterns

        new_response = request.form.get(f'response_{intent["tag"]}')
        if new_response is not None:  # Check if response field is not empty
            intent['responses'] = [new_response]

    with open(DATASET_PATH, 'w') as json_data:
        json.dump(updated_intents, json_data, indent=4)

    return redirect(url_for('edit_responses'))


# Add route to handle adding new intents
@app.route('/add_intent', methods=['POST'])
@login_required
def add_intent():
    # Get the data for the new intent from the request
    new_tag = request.form.get('tag')
    new_patterns = request.form.get('patterns')
    new_responses = request.form.get('responses')
    new_context_set = request.form.get('context_set')
    
    # Load existing intents
    intents, _, _, _, _ = load_data_and_model()

    # Append new intent to the list
    new_intent = {
        "tag": new_tag,
        "patterns": [pattern.strip() for pattern in new_patterns.split('\n') if pattern.strip()],
        "responses": [response.strip() for response in new_responses.split('\n') if response.strip()],
        "context_set": new_context_set
    }
    intents['intents'].append(new_intent)

    # Save updated intents to dataset.json
    with open(DATASET_PATH, 'w') as json_data:
        json.dump(intents, json_data, indent=4)

    # Redirect back to the admin page
    return redirect(url_for('add_intent_page'))

@app.route('/list_files', methods=['GET'])
@login_required
def list_files():
    upload_folder = app.config['UPLOAD_FOLDER']
    files = [f for f in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, f))]
    file_urls = [url_for('static', filename=f'uploads/{file}', _external=True) for file in files]
    return jsonify({'files': file_urls})


@app.route('/delete_files', methods=['POST'])
@login_required
def delete_files():
    data = request.json
    file_urls = data.get('file_urls')
    
    if not file_urls:
        return jsonify({'error': 'No files provided'}), 400
    
    for file_url in file_urls:
        filename = file_url.split('/')[-1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
    
    return jsonify({'success': True})


@app.route('/manage_files')
@login_required
def manage_files():
    return render_template('manage_files.html')



# Route to render the page for adding a new intent
@app.route('/add_intent_page')
@login_required
def add_intent_page():
    return render_template('add_intent.html')


@app.route('/edit_responses')
@login_required
def edit_responses():
    intents, _, _, _, _ = load_data_and_model()
    return render_template('edit_responses.html', intents=intents['intents'])


@app.route('/train_model_page', methods=['GET'])
@login_required
def train_model_page():
        return render_template('train.html')



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        file_url = url_for('static', filename=f'uploads/{filename}')
        mime_type, _ = mimetypes.guess_type(filename)
        
        if mime_type == 'application/pdf':
            tag = f'<embed src="{file_url}" style="width: 285px;height:450px;"></embed>'
        elif mime_type.startswith('image'):
            tag = f'<img src="{file_url}" style="width: 285px;height:450px;">'
        elif mime_type == 'audio/mpeg':
            tag = f'<audio controls src="{file_url}" style="width: 285px;"></audio>'
        elif mime_type == 'video/mp4':
            tag = f'<video controls src="{file_url}" style="width: 285px;height:450px;"></video>'
        else:
            tag = f'Unsupported file format: {filename}'
        
        return jsonify({'tag': tag})
    else:
        flash('File type not allowed')
        return redirect(request.url)



@app.route('/train_model', methods=['POST'])
@login_required
def train_model():
    # Ensure you set the correct path to your training script
    os.system('python NeuralNet\\Training\\train.py')
    return redirect(url_for('admin'))

# Add a route to provide training progress
@app.route('/get_training_progress', methods=['GET'])
def get_training_progress():
    try:
        with open('training_progress.txt', 'r') as file:
            progress = float(file.read())
        return jsonify({'progress': progress})
    except FileNotFoundError:
        abort(404)




# Function to check if a username and password are valid
def is_valid_credentials(username, password):
    return users.get(username) == password

# Route to handle login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if is_valid_credentials(username, password):
            session['username'] = username
            return redirect(url_for('admin'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

# Route to handle logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/delete_intent_page')
@login_required
def delete_intents_page():
    intents, _, _, _, _ = load_data_and_model()
    return render_template('delete_intent.html', intents=intents['intents'])

@app.route('/delete_intents', methods=['POST'])
@login_required
def delete_intents():
    intents_to_delete = request.form.getlist('intents_to_delete[]')
    if not intents_to_delete:
        abort(400, description="Intents to delete not provided")

    intents, _, _, _, _ = load_data_and_model()
    updated_intents = [intent for intent in intents['intents'] if intent['tag'] not in intents_to_delete]

    with open(DATASET_PATH, 'w') as json_data:
        json.dump({"intents": updated_intents}, json_data, indent=4)

    return redirect(url_for('admin'))



if __name__ == '__main__':
    app.run(debug=True)

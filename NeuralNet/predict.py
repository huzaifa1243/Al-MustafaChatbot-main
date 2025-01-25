import json
import random
import torch
from model import NeuralNet
from utils import tokenize, stem, bag_of_words

def load_data_and_model():
    # Load intents data from a JSON file
    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    # Load the pre-trained model and data
    FILE = "data.pth"
    data = torch.load(FILE)

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

    return intents, all_words, tags, model, device

def chatbot_response(intents, all_words, tags, model, device, sentence):
    bot_name = "Sam"
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
                return f"{bot_name}: {random.choice(intent['responses'])}"
    else:
        return f"{bot_name}: I do not understand..."

def run_chatbot():
    intents, all_words, tags, model, device = load_data_and_model()

    print("Let's chat! (type 'quit' to exit)")

    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        response = chatbot_response(intents, all_words, tags, model, device, sentence)
        print(response)

if __name__ == "__main__":
    run_chatbot()

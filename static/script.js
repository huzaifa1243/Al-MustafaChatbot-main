// Function to fetch predicted response from backend
function getPredictedResponse() {
    const question = document.getElementById('question').value; // Get the user's question
    const chatHistory = document.getElementById('chat-history');

    // Append user's message to the chat history
    appendMessage(question, 'user');

    fetch('http://127.0.0.1:5000/get_response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: question }) // Send the question to the backend as JSON
    })
    .then(response => response.json())
    .then(data => {
        // Extract predicted response from data
        const predictedResponse = data.response;

        // Append the predicted response to the chat history
        appendMessage(predictedResponse, 'response');
    })
    .catch(error => console.error('Error:', error));

    // Clear the input field
    document.getElementById('question').value = '';
}

// Function to append a message to the chat history
// Function to append a message to the chat history
function appendMessage(message, sender) {
    const chatHistory = document.getElementById('chat-history');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    
    // Create image element for avatar
    const avatar = document.createElement('img');
    avatar.classList.add('avatar');
    avatar.src = sender === 'user' ? '/static/sender.jpeg' : '/static/sender.jpeg'; // Adjust path based on your project structure

    // Append avatar to message element
    messageElement.appendChild(avatar);

    // Append message text to message element
    const messageText = document.createElement('div');
    messageText.innerText = message;
    messageElement.appendChild(messageText);

    // Append message element to chat history
    chatHistory.appendChild(messageElement);

    // Automatically scroll to the bottom of the chat history
    messageElement.scrollIntoView();
}

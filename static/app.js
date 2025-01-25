class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.chatbox__send--footer'),
            inputField: document.querySelector('.chatbox__footer input'),
            darkModeToggle: document.getElementById('darkModeToggle')
        }

        // Define the audio object and load the audio file
        this.audio = new Audio('/static/audio/tip_small.mp3');

        this.state = false;
        this.messages = [];
    }

    display() {
        const { openButton, chatBox, sendButton, inputField, darkModeToggle } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox, inputField));
        inputField.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox, inputField);
            }
        });

        darkModeToggle.addEventListener('click', () => this.toggleDarkMode());
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // Show or hide the box
        if (this.state) {
            chatbox.classList.add('chatbox--active');
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    toggleDarkMode() {
        document.body.classList.toggle('dark-mode');
        document.querySelector('.chatbox__support').classList.toggle('dark-mode');
        document.querySelector('.chatbox__header').classList.toggle('dark-mode');
        document.querySelector('.chatbox__messages').classList.toggle('dark-mode');
        document.querySelectorAll('.messages__item').forEach(item => item.classList.toggle('dark-mode'));
        document.querySelector('.chatbox__footer').classList.toggle('dark-mode');
        document.querySelector('.chatbox__footer input').classList.toggle('dark-mode');
        document.querySelector('.chatbox__button button').classList.toggle('dark-mode');
        document.querySelector('.dark-mode-toggle').classList.toggle('dark-mode');
    }

    onSendButton(chatbox, inputField) {
        var text1 = inputField.value.trim();
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);

        fetch('/get_response', {
            method: 'POST',
            body: JSON.stringify({ question: text1 }),
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(response => response.json())
        .then(data => {
            let msg2 = { name: "Sam", message: data.response };
            this.messages.push(msg2);
            this.updateChatText(chatbox);
            inputField.value = '';

            // Play the message tone
            this.playMessageTone();
        })
        .catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox);
            inputField.value = '';
        });
    }

    playMessageTone() {
        this.audio.play();
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "Sam") {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;

        // Apply dark mode to new messages if dark mode is active
        if (document.body.classList.contains('dark-mode')) {
            chatmessage.querySelectorAll('.messages__item').forEach(item => item.classList.add('dark-mode'));
        }
    }
}

const chatbox = new Chatbox();
chatbox.display();

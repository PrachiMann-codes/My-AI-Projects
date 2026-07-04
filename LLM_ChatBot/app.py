from flask import Flask, request, render_template
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    try:
        data = request.get_json()  # safer than get_data
        input_text = data['prompt']

        # Create conversation history string
        history = " ".join(conversation_history)

        # Combine history + new input
        full_input = history + " " + input_text if history else input_text

        # Tokenize
        inputs = tokenizer(full_input, return_tensors="pt")

        # Generate response
        outputs = model.generate(**inputs, max_length=60)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Update history
        conversation_history.append(input_text)
        conversation_history.append(response)

        return response  # Flask will return this as plain text
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run()

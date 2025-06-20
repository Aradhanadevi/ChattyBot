from flask import Flask, request, jsonify
import pickle
import os

# Load model and vectorizer
with open('chat_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'reply': "Please say something 🥲"})

    # Transform and predict
    X = vectorizer.transform([user_message])
    prediction = model.predict(X)[0]

    return jsonify({'reply': prediction})

# ✅ Only run ONCE with dynamic port (for Render)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

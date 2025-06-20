from flask import Flask, request, make_response
import pickle
import os
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')  # 👈 Ensures server uses UTF-8

# Load model pipeline (includes vectorizer + model)
with open('chat_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        reply = "Please say something 🥲"
    else:
        reply = model.predict([user_message])[0]

    # ✅ Ensure UTF-8 JSON with emojis
    response = make_response(json.dumps({'reply': reply}, ensure_ascii=False))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

# ✅ Dynamic port for Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

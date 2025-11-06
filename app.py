from flask import Flask, render_template, request, jsonify
import joblib
import random

app = Flask(__name__)

# Load saved model components
model = joblib.load("models/health_chatbot_model.pkl")
vectorizer = joblib.load("models/health_chatbot_vectorizer.pkl")
le = joblib.load("models/health_chatbot_labelencoder.pkl")

# Load dataset for responses
import pandas as pd
data = pd.read_csv("health_chatbot_dataset.csv")

# Chatbot response function
def chatbot_response(user_input):
    X = vectorizer.transform([user_input])
    intent_pred = model.predict(X)[0]
    intent_name = le.inverse_transform([intent_pred])[0]
    responses = data[data["intent"] == intent_name]["response"].values
    if len(responses) > 0:
        return random.choice(responses)
    return "I'm not sure, but try to stay healthy!"

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["message"]
    bot_reply = chatbot_response(user_input)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)

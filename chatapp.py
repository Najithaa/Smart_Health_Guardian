from flask import Flask, render_template, request, jsonify
import difflib

app = Flask(__name__)

# Predefined health-related responses
responses = {
    "hi": "Hello! How can I assist you with your health today?",
    "hello": "Hi there! What health-related query do you have?",
    "how are you": "I'm just a bot, but I'm here to help you with health-related questions!",
    "what is fever": "Fever is a temporary increase in body temperature, often due to an illness.",
    "symptoms of covid": "Common symptoms include fever, cough, shortness of breath, fatigue, and loss of taste or smell.",
    "how to reduce fever": "You can take paracetamol, drink plenty of fluids, and get rest. If symptoms persist, consult a doctor.",
    "healthy diet": "A healthy diet includes fruits, vegetables, whole grains, lean proteins, and plenty of water.",
    "exercise benefits": "Regular exercise improves heart health, boosts immunity, and helps maintain a healthy weight.",
    "mental health tips": "Practice mindfulness, get enough sleep, stay active, and talk to someone if you feel overwhelmed.",
    "headache remedies": "Try drinking water, resting, reducing screen time, or taking a mild pain reliever if needed.",
}

def get_best_match(query):
    query = query.lower()
    best_match = difflib.get_close_matches(query, responses.keys(), n=1, cutoff=0.5)
    return responses.get(best_match[0], "I'm not sure about that. Can you try asking in a different way?") if best_match else "I don't have an answer for that yet."

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.form["msg"]
    response = get_best_match(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
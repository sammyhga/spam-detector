import os
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the vectorizer and model
vectorizer_path = os.path.join("models", "vectorizer.pkl")
model_path = os.path.join("models", "model.pkl")

with open(vectorizer_path, "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        user_message = request.form["message"]
        
        # Transform the message using the vectorizer
        message_vector = vectorizer.transform([user_message])
        
        # Get probabilities
        probabilities = model.predict_proba(message_vector)[0]
        spam_probability = probabilities[1]  # Assuming spam is class 1
        spam_percentage = spam_probability * 100  # Convert to percentage
        
        if spam_percentage > 50:  # If greater than 50%, consider it spam
            result = f"Spam ({spam_percentage:.2f}%)"
        else:
            result = "No Spam"
        
        return render_template("index.html", result=result)
    
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

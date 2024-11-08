import os
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the vectorizer and model
with open(os.path.join('models', 'vectorizer.pkl'), 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
with open(os.path.join('models', 'model.pkl'), 'rb') as model_file:
    model = pickle.load(model_file)

def detect_spam(message):
    # Transform the message using the vectorizer
    vectorized_message = vectorizer.transform([message])
    # Predict using the loaded model
    prediction = model.predict(vectorized_message)
    # Return "Spam" or "Not Spam" based on the prediction
    return "Spam" if prediction[0] == 1 else "Not Spam"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        message = request.form['message']
        # Detect spam or not spam
        result = detect_spam(message)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

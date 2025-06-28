from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route('/')
def home():
    return "✅ Sentiment Analysis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Please provide text"}), 400
    
    text = data['text']
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    result = "Positive" if prediction == 1 else "Negative"

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model, vectorizer, and label encoder only once to optimize
model = joblib.load("models/naive_bayes_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Function to predict genre from description using the loaded model
def predict_genre(description):
    description_vectorized = vectorizer.transform([description])
    genre_encoded = model.predict(description_vectorized)[0]
    genre = label_encoder.inverse_transform([genre_encoded])[0]
    return genre

@app.route('/')
def home():
    return render_template('index.html')  # Renders the HTML template

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({'error': 'No description provided'}), 400

    description = data['description']
    genre = predict_genre(description)  # Predict genre using the function
    return jsonify({'predicted_genre': genre})

if __name__ == '__main__':
    app.run(debug=True)

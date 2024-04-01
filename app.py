# app.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json 

app = Flask(__name__)

# Load your trained machine learning model

model = joblib.load("./mle-intv-main/models/model.pkl")  # Replace "your_model.pkl" with the path to your trained model file

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()
   
    # Preprocess the input data (assuming it's a list of features)
    features = preprocess(data)

    # Make prediction using the model
    prediction = model.predict(features)

    # Return the prediction
    return jsonify({'predictions': prediction.tolist()})

def preprocess(data):

    # Parse the JSON string into a Python data structure
    data_dict = json.loads(data)
    
    # Convert the data into a DataFrame
    features = pd.DataFrame(data_dict)
    return features

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

import csv
import requests
import pandas as pd

# Define the path to your CSV file
csv_file = './data/score.csv'

df = pd.read_csv(csv_file)

# URL of the Flask server endpoint
url = "http://localhost:5000/predict"


# Prepare the input data
input_data = df.to_json(orient='records')

# Send a POST request with the input data to the server endpoint
response = requests.post(url, json=input_data)

# Check if the request was successful (status code 200)
if response.status_code == 200:

    # Print the prediction result
    print("Prediction:", response.json())
    print()
else:
    print("Error:", response.text)


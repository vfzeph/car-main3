import os
import joblib
import pandas as pd
import traceback
from flask import Flask, request, jsonify, render_template
import traceback

app = Flask(__name__)

# Directory paths
base_dir = os.path.dirname(__file__)
models_dir = os.path.join(base_dir, "model_outputs")
encoders_dir = os.path.join(base_dir, "model_outputs", "label_encoders")
scaler_path = os.path.join(models_dir, "scaler.pkl")

# Load all models
models = {}
for file_name in os.listdir(models_dir):
    if file_name.endswith('.pkl') and file_name != 'scaler.pkl':
        model_name = file_name.split('.')[0]
        models[model_name] = joblib.load(os.path.join(models_dir, file_name))

# Load the scaler, with error handling if it doesn't exist
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    print(f"Scaler file not found at path {scaler_path}. Make sure it's been saved correctly.")
    scaler = None

# Load all label encoders
label_encoders = {}
for file_name in os.listdir(encoders_dir):
    if file_name.endswith('.pkl'):
        feature_name = file_name.split('.')[0]
        label_encoders[feature_name] = joblib.load(os.path.join(encoders_dir, file_name))

def preprocess_data(data, label_encoders, scaler):
    # Apply label encoders to each relevant column
    for feature, encoder in label_encoders.items():
        if feature in data.columns:
            data[feature] = encoder.transform(data[feature].astype(str))
    
    # Transform the data using the loaded scaler
    if scaler is not None:
        scaled_data = scaler.transform(data)
        return scaled_data
    else:
        return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if scaler is None:
        return jsonify({"error": "Scaler not loaded"}), 500

    try:
        file = request.files['file']
        if not file:
            return jsonify({"error": "No file provided"}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Please upload a CSV file"}), 400

        data = pd.read_csv(file)
        processed_data = preprocess_data(data, label_encoders, scaler)

        all_predictions = {}
        for model_name, model in models.items():
            predictions = model.predict(processed_data)
            all_predictions[model_name] = predictions.tolist()

        return jsonify({"all_predictions": all_predictions})
    except Exception as e:
        # This will print the full traceback to the console
        traceback.print_exc()
        # Return the error message in the response
        return jsonify({"error": str(e)}), 500  # Ensure this line is properly indented

if __name__ == '__main__':
    app.run(debug=True)  # This line should be properly indented as well

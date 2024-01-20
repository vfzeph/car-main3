import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from fit_models import prepare_data

app = Flask(__name__)
app.secret_key = 'your_secret_key'

import matplotlib
matplotlib.use('Agg')

# Directory paths
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "model_outputs")

# Load scaler and models
models = {f.split('.')[0]: joblib.load(os.path.join(models_dir, f))
          for f in os.listdir(models_dir) if f.endswith('.pkl') and f != 'scaler.pkl'}

def preprocess_data(data):
    features, _, _ = prepare_data(data)
    return features

def make_predictions(processed_data):
    predictions = {model_name: model.predict(processed_data)
                   for model_name, model in models.items()}
    return predictions

def generate_confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    plt.savefig(os.path.join(models_dir, 'conf_matrix.png'), format='png', bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
    return os.path.join(models_dir, 'conf_matrix.png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            try:
                data = pd.read_csv(file)
                predictions, conf_matrix_plots = process_data_and_predict(data)
                return render_template('results.html', predictions=predictions, conf_matrix_plots=conf_matrix_plots)
            except Exception as e:
                flash(f"An error occurred: {e}")
                return redirect(url_for('index'))
        else:
            flash("Invalid file. Please upload a CSV file.")
            return redirect(url_for('index'))

    return redirect(url_for('index'))

def process_data_and_predict(data):
    """
    Process the data, train models, make predictions, and generate confusion matrix plots.
    """
    X, y, _ = prepare_data(data)  # Data preparation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    predictions = make_predictions(X_test)
    conf_matrix_plots = generate_confusion_matrix_plots(y_test, predictions)
    return predictions, conf_matrix_plots

def generate_confusion_matrix_plots(y_test, predictions):
    """
    Generate confusion matrix plots for each model.
    """
    conf_matrix_plots = {}
    for model_name, model_pred in predictions.items():
        plot_path = os.path.join(models_dir, f"{model_name}_confusion_matrix.png")
        plt.figure()  # Create a new figure
        generate_confusion_matrix_plot(y_test, model_pred)
        conf_matrix_plots[model_name] = f"{model_name}_confusion_matrix.png"

    return conf_matrix_plots

@app.route('/model_outputs/<filename>')
def model_output(filename):
    return send_from_directory(models_dir, filename)

if __name__ == '__main__':
    app.run(port=5002, debug=True)

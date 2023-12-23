from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

app = Flask(__name__, template_folder='template')

# Load the KNN model
model_path_knn = "model/knn_model.joblib"
knn = joblib.load(model_path_knn)

# Load the Linear Regression model
lr_model = joblib.load("model/linear_regression_model.joblib")

# Load the LSTM model
lstm_model = load_model('model/lstm_model.h5')
lstm_model.compile(run_eagerly=True)
scaler = MinMaxScaler()

def predict_knn(features):
    input_data_knn = pd.DataFrame([features], columns=['High', 'Low', 'Open_Price'])
    prediction_knn = knn.predict(input_data_knn)
    return prediction_knn[0]

def predict_lr(stock_price):
    input_df_lr = pd.DataFrame({'Stock Price': [stock_price]})
    imputer = SimpleImputer(strategy='mean')
    input_df_lr_no_nan = imputer.fit_transform(input_df_lr)
    prediction_lr = lr_model.predict(input_df_lr_no_nan)
    return prediction_lr[0]

def predict_lstm(start_date, end_date):
    tf.config.run_functions_eagerly(False)

    num_days = (end_date - start_date).days + 1
    input_data = np.arange(num_days).reshape(-1, 1)
    normalized_input_data = scaler.fit_transform(input_data)
    predicted_values = lstm_model.predict(normalized_input_data)

    tf.config.run_functions_eagerly(True)

    denormalized_predictions = scaler.inverse_transform(predicted_values)

    return denormalized_predictions.flatten().tolist()

def generate_date_range(start_date, end_date):
    date_range = [str(start_date + timedelta(days=i)) for i in range((end_date-start_date).days + 1)]
    return date_range

@app.route('/')
def home():
    return render_template('navigation.html')

# KNN Routes
@app.route('/knn')
def knn_home():
    return render_template('index1.html')

@app.route('/predict_knn', methods=['POST'])
def predict_knn_route():
    try:
        features_knn = [
            float(request.form['High']),
            float(request.form['Low']),
            float(request.form['Open_Price']),
        ]

        prediction_knn = predict_knn(features_knn)

        return render_template('index1.html', prediction=prediction_knn)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template('index1.html', error=error_message)

# Linear Regression Routes
@app.route('/linear_regression')
def linear_regression_home():
    return render_template('index3.html')

@app.route('/predict_lr', methods=['POST'])
def predict_lr_route():
    try:
        stock_price = float(request.form['Stock_Price'].replace(',', ''))
        prediction_lr = predict_lr(stock_price)

        return render_template('index3.html', prediction_lr=prediction_lr)

    except Exception as e:
        app.logger.error(f"Exception: {str(e)}")
        return render_template('index3.html', error_lr="Error occurred. Please check your input.")

# LSTM Routes
@app.route('/lstm')
def lstm_home():
    return render_template('index.html')

@app.route('/predict_lstm', methods=['POST'])
def predict_lstm_route():
    try:
        start_date_str = request.form['start_date']
        end_date_str = request.form['end_date']

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        predicted_values = predict_lstm(start_date, end_date)
        date_range = generate_date_range(start_date, end_date)

        return jsonify({'dates': date_range, 'values': predicted_values})

    except Exception as e:
        app.logger.error(f"Exception: {str(e)}")
        return jsonify({'error': 'Error occurred. Please check your input.'})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

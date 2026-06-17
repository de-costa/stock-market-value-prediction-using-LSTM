from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

app = Flask(__name__)

# load model + scaler
model = load_model("stock-market.keras")
scaler = joblib.load("scaler.pkl")

WINDOW_SIZE = 50

def prepare_input(data):
    data = np.array(data).reshape(-1, 1)
    scaled = scaler.transform(data)
    return np.array([scaled])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = request.form.get('values')

        
        values = [float(x) for x in values.split(',')]

        if len(values) != WINDOW_SIZE:
            return f"Please enter exactly {WINDOW_SIZE} values"

        input_data = prepare_input(values)

        prediction = model.predict(input_data)
        predicted_price = scaler.inverse_transform(prediction)

        return render_template(
            "index.html",
        prediction_text=f"Predicted Open Price: {predicted_price[0][0]:.2f}"
)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
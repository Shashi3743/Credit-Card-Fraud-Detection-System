from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Only these features are scaled
scale_features = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
threshold = 0.8

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        data = [float(request.form.get(feat)) for feat in [
            'distance_from_home',
            'distance_from_last_transaction',
            'ratio_to_median_purchase_price',
            'repeat_retailer',
            'used_chip',
            'used_pin_number',
            'online_order'
        ]]

        # Scale first 3 features
        scaled = scaler.transform([data[:3]])[0]
        full_data = np.array([*scaled, *data[3:]]).reshape(1, -1)

        # Predict probability
        prob = model.predict_proba(full_data)[0][1]
        prediction = "Fraud" if prob >= threshold else "Not Fraud"
        score = round(prob, 2)

        return render_template('result.html', prediction=prediction, score=score)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)

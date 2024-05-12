import numpy as np
from flask import Flask, render_template, request
import joblib


# Load the pre-trained machine learning model
model = joblib.load('C:/Users/ZAHRA/Documents/BSCS/calorieAI/trained_model.joblib')
app = Flask(__name__, static_url_path='/static')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])
        gender = float(request.form['gender'])
        
        # Create a numpy array for the input features
        input_features = np.array([[age, height, weight, duration, heart_rate, body_temp, gender]])
        
        # Make a prediction using the loaded model
        prediction = model.predict(input_features)
        
        # Render the prediction on the result page
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

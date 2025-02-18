from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the pre-trained model
model = pickle.load(open('breast_cancer_model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        radius = float(request.form['radius'])
        texture = float(request.form['texture'])
        perimeter = float(request.form['perimeter'])
        area = float(request.form['area'])
        smoothness = float(request.form['smoothness'])
        
        # Make prediction using the model
        prediction = model.predict([[radius, texture, perimeter, area, smoothness]])[0]
        
        # Map prediction to result
        prediction_map = {0: 'malignant', 1: 'benign'}
        result = prediction_map[prediction]
        
        # Set image URL based on prediction
        image_map = {
            'malignant': '/static/malignant.png',
            'benign': '/static/benign.png'
        }
        breast_image = image_map[result]

        return render_template('home.html', prediction_text=f'The predicted result is: {result}', breast_image=breast_image)
    
    except Exception as e:
        # Handle errors (e.g., invalid input)
        return render_template('home.html', prediction_text='Error: Invalid input data. Please try again.', breast_image=None)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

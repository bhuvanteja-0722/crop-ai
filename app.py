# app.py

from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained machine learning model
try:
    with open('crop_recommendation.model', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: Model file not found. Please run train.py first to create it.")
    exit()

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    if request.method == 'POST':
        # Get the data from the form
        try:
            n = float(request.form['nitrogen'])
            p = float(request.form['phosphorus'])
            k = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Create a numpy array for the model
            user_input = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
            
            # Make a prediction
            prediction = model.predict(user_input)
            
            # Format the prediction text
            prediction_text = f"The recommended crop is: {prediction[0].capitalize()}"

        except ValueError:
            prediction_text = "Invalid input. Please enter valid numbers."
        except Exception as e:
            prediction_text = f"An error occurred: {e}"

    # Render the HTML template with the prediction result
    return render_template('index.html', result=prediction_text)

# Run the app
if __name__ == '__main__':
    app.run(debug=True) # debug=True helps in development
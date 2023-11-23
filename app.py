from flask import Flask, request, render_template
import pickle

# Load the model from the saved pickle file
with open('diab.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a Flask app
app = Flask(__name__)

# Define a route for the web interface
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    pregnancies = int(request.form['pregnancies'])
    glucose = int(request.form['glucose'])
    blood_pressure = int(request.form['blood_pressure'])
    skin_thickness = int(request.form['skin_thickness'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])

    # Make a prediction using the model
    prediction = model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf]])

    # Return the prediction as a response
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
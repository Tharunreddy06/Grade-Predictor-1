from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully!")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    prediction_color = None

    if request.method == 'POST':
        try:
            # Get input values from the form
            studytime = float(request.form['studytime'])
            failures = int(request.form['failures'])
            absences = int(request.form['absences'])
            G1 = float(request.form['G1'])
            G2 = float(request.form['G2'])

            # Prepare input as NumPy array
            user_input = np.array([[studytime, failures, absences, G1, G2]])

            # Make prediction
            predicted_grade = model.predict(user_input)[0]

            # Determine color based on predicted grade
            if predicted_grade < 10:
                prediction_color = 'red'
            else:
                prediction_color = 'green'

            prediction = f"Predicted grade: {predicted_grade}"
        except Exception as e:
            prediction = "There was an error in prediction."
            prediction_color = 'yellow'  # Or handle the error as you wish

    return render_template('index.html', prediction=prediction, prediction_color=prediction_color)

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000, debug=True)

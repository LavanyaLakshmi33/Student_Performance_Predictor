from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    sports = float(request.form['sports'])
    attendance = float(request.form['attendance'])

    prediction = model.predict([[hours, sports, attendance]])
    result = "Pass" if prediction[0] == 1 else "Fail"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

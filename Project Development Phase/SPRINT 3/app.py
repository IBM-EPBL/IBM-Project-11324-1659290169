import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    print(prediction)
    if(output == "Absence"):
        return render_template('absence.html')
    else:
        return render_template('presence.html')

@app.route('/homee', methods=['GET', 'POST'])
def homee():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(debug=True)

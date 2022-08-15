from flask import Flask, redirect, url_for, render_template, request
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
model = pickle.load(open("finalized_model.pkl", 'rb'))
app = Flask(__name__)
global __predict
__predict = None

@app.route("/")
def home():
    return render_template("index.html")
global usr
usr=0
@app.route("/", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        Pregnancies = request.form["Pregnancies"]
        print(Pregnancies)
        Glucose= request.form["Glucose"]
        BloodPressure= request.form["BloodPressure"]
        SkinThickness= request.form["SkinThickness"]
        Insulin= request.form["Insulin"]
        BMI= request.form["BMI"]
        DiabetesPedigreeFunction= request.form["DiabetesPedigreeFunction"]
        Age= request.form["Age"]
        inputa = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
        inputa = np.asarray(inputa)


        inputa = inputa.reshape(1, -1)
        scaler = StandardScaler()
        scaler.fit(inputa)
        inputa = scaler.transform(inputa)
        __predict = model.predict(inputa)
        if (__predict == [0]):
            __predict = 0
        else:
            __predict=1
        return render_template("index.html", usr=__predict)

if __name__ == "__main__":
    app.run(debug=True)
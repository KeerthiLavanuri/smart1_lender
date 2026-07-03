import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Current folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
model = pickle.load(open(os.path.join(BASE_DIR, "rdf.pkl"), "rb"))

# Load scaler
scale = pickle.load(open(os.path.join(BASE_DIR, "scale1.pkl"), "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():

    Gender = int(request.form["Gender"])
    Married = int(request.form["Married"])
    Dependents = int(request.form["Dependents"])
    Education = int(request.form["Education"])
    Self_Employed = int(request.form["Self_Employed"])
    ApplicantIncome = float(request.form["ApplicantIncome"])
    CoapplicantIncome = float(request.form["CoapplicantIncome"])
    LoanAmount = float(request.form["LoanAmount"])
    Loan_Amount_Term = float(request.form["Loan_Amount_Term"])
    Credit_History = float(request.form["Credit_History"])
    Property_Area = int(request.form["Property_Area"])

    data = [[
        Gender,
        Married,
        Dependents,
        Education,
        Self_Employed,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        Property_Area
    ]]

    data = scale.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Loan Will Be Approved"
    else:
        result = "Loan Will Not Be Approved"

    return render_template("submit.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
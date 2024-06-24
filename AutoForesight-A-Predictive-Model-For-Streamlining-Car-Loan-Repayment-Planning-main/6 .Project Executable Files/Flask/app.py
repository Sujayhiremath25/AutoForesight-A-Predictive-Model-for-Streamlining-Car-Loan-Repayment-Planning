from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("knn_model_pickle.pkl", "rb"))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    client_id = request.form.get('client_id')
    client_income = request.form.get('client_income')
    car_owner = request.form.get('car_owner')
    bike_owned = request.form.get('bike_owned')
    active_loan = request.form.get('active_loan')
    credit_amount = request.form.get('credit_amount')
    loan_annuity = request.form.get('loan_annuity')
    client_income_type = request.form.get('client_income_type')
    client_education = request.form.get('client_education')
    client_marital_status = request.form.get('client_marital_status')
    client_gender = request.form.get('client_gender')
    loan_contract_type = request.form.get('loan_contract_type')
    client_occupation = request.form.get('client_occupation')
    type_organization = request.form.get('type_organization')

    data = [
        client_id, client_income, car_owner,
        bike_owned, active_loan, credit_amount,
        loan_annuity, client_income_type,
        client_education,
        client_marital_status,
        client_gender, loan_contract_type,
        client_occupation,
        type_organization,
    ]
    

    # Perform prediction or processing here (not implemented in this example)
    data = [int(x) for x in data]
    final_features = [np.array(data)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 1)

    # Render the form again with submitted values and prediction result
    return render_template('index.html', 
                           client_id=client_id, client_income=client_income,
                           car_owner=car_owner, bike_owned=bike_owned,
                           active_loan=active_loan, credit_amount=credit_amount,
                           loan_annuity=loan_annuity, client_income_type=client_income_type,
                           client_education=client_education, client_marital_status=client_marital_status,
                           client_gender=client_gender, loan_contract_type=loan_contract_type,
                           client_occupation=client_occupation, type_organization=type_organization,
                           prediction_text="Result: {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)

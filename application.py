from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)


column_transformer = pickle.load(open("models/column_transformer.pkl", "rb"))
model = pickle.load(open("models/best_rf.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [
            request.form.get('gender'),
            int(request.form.get('SeniorCitizen')),
            request.form.get('Partner'),
            request.form.get('Dependents'),
            request.form.get('PhoneService'),
            request.form.get('MultipleLines'),
            request.form.get('InternetService'),
            request.form.get('OnlineSecurity'),
            request.form.get('OnlineBackup'),
            request.form.get('DeviceProtection'),
            request.form.get('TechSupport'),
            request.form.get('StreamingTV'),
            request.form.get('StreamingMovies'),
            request.form.get('Contract'),
            request.form.get('PaperlessBilling'),
            request.form.get('PaymentMethod'),
            request.form.get('tenure_group'),
            float(request.form.get('MonthlyCharges')),
            float(request.form.get('TotalCharges'))
        ]
        
       
        column_names = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
            "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group", 
            "MonthlyCharges", "TotalCharges"
        ]

        
        input_df = pd.DataFrame([input_data], columns=column_names)
        print("Converted Input DataFrame:\n", input_df)

       
        transformed_input = column_transformer.transform(input_df)
        print("Transformed Input Data:", transformed_input)

        
        prediction = model.predict_proba(transformed_input)[0][1]
        print("Prediction Probability:", prediction)

        result = f"🔴 High Risk of Churn ({round(prediction * 100, 2)}%)" if prediction > 0.5 else f"🟢 Low Risk of Churn ({round(prediction * 100, 2)}%)"
        
        return jsonify({'prediction': result})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': f"Error processing request: {e}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns a dynamic port
    app.run(host="0.0.0.0", port=port)

from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained machine learning model
with open("C:\\Users\\irosh\\strokepredectionapp\\model\\stroke_prediction_model.pkl", 'rb') as model_file:
    blood_pressure_model = pickle.load(model_file)

def predict_blood_pressure(age, systolic_pressure, diastolic_pressure, hypertension, gender_encoded, never_smoked, smokes):
    try:
        # Prepare the input data for prediction
        input_data = [[age, systolic_pressure, diastolic_pressure, hypertension, gender_encoded, never_smoked, smokes]]

        # Use the loaded model to make predictions
        prediction = blood_pressure_model.predict(input_data)

        # Interpret the prediction result
        if prediction[0] == 1:
            return "High Risk of Stroke"
        else:
            return "Low Risk of Stroke"

    except Exception as e:
        return f"Prediction Error: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def home():
    error_msg = None
    result = None

    if request.method == 'POST':
        try:
            # Extract form data
            age = float(request.form['age'])
            systolic_pressure = float(request.form['systolic_pressure'])
            diastolic_pressure = float(request.form['diastolic_pressure'])
            hypertension = int(request.form['hypertension'])
            gender_encoded = int(request.form['gender_encoded'])
            never_smoked = int(request.form['never_smoked'])
            smokes = int(request.form['smokes'])

            # Perform predictions using the model
            result = predict_blood_pressure(age, systolic_pressure, diastolic_pressure, 
                                             hypertension, gender_encoded, never_smoked, smokes)

        except ValueError:
            error_msg = "Invalid input. Please enter valid numeric values for the features."
        except Exception as e:
            error_msg = f"Error: {str(e)}"

    return render_template('index.html', error=error_msg, result=result)

if __name__ == '__main__':
    app.run(debug=True)

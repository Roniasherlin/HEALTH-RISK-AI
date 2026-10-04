import os
import joblib
import pandas as pd


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "heart_risk_model.pkl"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# 3. PREDICTION FUNCTION
# ============================================================

def predict_risk(patient_data):

    # Convert input dictionary into DataFrame
    input_data = pd.DataFrame([patient_data])

    # Generate prediction
    prediction = model.predict(input_data)[0]

    # Generate probability
    probability = model.predict_proba(input_data)[0][1]

    # Convert prediction into readable result
    if prediction == 1:
        risk = "Higher Risk"
    else:
        risk = "Lower Risk"

    return {
        "prediction": int(prediction),
        "risk": risk,
        "probability": float(probability)
    }


# ============================================================
# 4. TEST THE MODEL
# ============================================================

if __name__ == "__main__":

    sample_patient = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 140,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }

    result = predict_risk(sample_patient)

    print("\n" + "=" * 50)
    print("HEALTHRISK AI - PREDICTION")
    print("=" * 50)

    print(f"Prediction : {result['risk']}")
    print(
        f"Risk Probability : "
        f"{result['probability'] * 100:.2f}%"
    )

    print("=" * 50)
import os
import joblib
import pandas as pd
import shap


# ============================================================
# 1. PROJECT PATHS
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
# 3. CREATE SHAP EXPLAINER
# ============================================================

# Our saved model is a Pipeline:
# Imputer → Random Forest
#
# SHAP needs to explain the Random Forest itself.

rf_model = model.named_steps["model"]

explainer = shap.TreeExplainer(rf_model)


# ============================================================
# 4. FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# ============================================================
# 5. EXPLAIN A SINGLE PREDICTION
# ============================================================

def explain_prediction(patient_data):

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame(
        [patient_data],
        columns=FEATURE_NAMES
    )

    # Apply the same preprocessing used during training
    preprocessor = model.named_steps["imputer"]

    processed_input = preprocessor.transform(input_df)

    # Generate SHAP values
    shap_values = explainer.shap_values(
        processed_input
    )

    # SHAP can return different structures
    # depending on the installed version.
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0, :, 1]

    # Create explanation table
    explanation_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "value": input_df.iloc[0].values,
        "shap_value": values
    })

    # Absolute contribution
    explanation_df["absolute_impact"] = (
        explanation_df["shap_value"].abs()
    )

    # Most influential features first
    explanation_df = explanation_df.sort_values(
        "absolute_impact",
        ascending=False
    )

    return explanation_df


# ============================================================
# 6. TEST THE EXPLANATION
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

    # Prediction
    input_df = pd.DataFrame(
        [sample_patient],
        columns=FEATURE_NAMES
    )

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        risk = "Higher Risk"
    else:
        risk = "Lower Risk"

    # SHAP explanation
    explanation = explain_prediction(
        sample_patient
    )

    print("\n" + "=" * 70)
    print("HEALTHRISK AI - SHAP EXPLANATION")
    print("=" * 70)

    print(f"Prediction : {risk}")
    print(
        f"Probability : "
        f"{probability * 100:.2f}%"
    )

    print("\nTop features influencing this prediction:")

    print(
        explanation[
            [
                "feature",
                "value",
                "shap_value"
            ]
        ].head(5).to_string(
            index=False
        )
    )

    print("=" * 70)
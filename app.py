import os
import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "heart_risk_model.pkl"
)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HealthRisk AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #888888;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .small-note {
        font-size: 13px;
        color: #888888;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# 5. FEATURE NAMES
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
# 6. HUMAN-READABLE FEATURE NAMES
# ============================================================

FEATURE_LABELS = {

    "age":
        "Age",

    "sex":
        "Sex",

    "cp":
        "Chest Pain Type",

    "trestbps":
        "Resting Blood Pressure",

    "chol":
        "Serum Cholesterol",

    "fbs":
        "Fasting Blood Sugar",

    "restecg":
        "Resting ECG",

    "thalach":
        "Maximum Heart Rate",

    "exang":
        "Exercise-Induced Angina",

    "oldpeak":
        "ST Depression",

    "slope":
        "ST Segment Slope",

    "ca":
        "Major Vessels",

    "thal":
        "Thalassemia Measurement"
}


# ============================================================
# 7. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">❤️ HealthRisk AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Machine Learning Risk Prediction & Explainable AI
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    """
    HealthRisk AI demonstrates an end-to-end machine-learning
    workflow using the UCI Heart Disease dataset. The application
    predicts the model's positive-class risk and uses SHAP to
    explain which input features influenced an individual prediction.
    """
)

st.divider()


# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.title("❤️ HealthRisk AI")

st.sidebar.subheader("Project Pipeline")

st.sidebar.markdown(
    """
    🧹 **Data Preprocessing**

    🤖 **Model Training**

    ⚙️ **Hyperparameter Tuning**

    🔮 **Risk Prediction**

    🧠 **SHAP Explainability**
    """
)

st.sidebar.divider()

st.sidebar.subheader("Model Information")

st.sidebar.write(
    """
    **Algorithm**

    Tuned Random Forest

    **Dataset**

    UCI Heart Disease

    **Training Samples**

    242

    **Test Samples**

    61
    """
)

st.sidebar.divider()

st.sidebar.warning(
    """
    ⚠️ Educational Project Only

    This application is not a medical diagnostic tool and
    should not be used for medical decisions.
    """
)


# ============================================================
# 9. PATIENT INFORMATION
# ============================================================

st.header("🧑 Patient Information")

st.write(
    "Enter the parameters below and click **Analyze Risk**."
)

col1, col2, col3 = st.columns(3)


# ============================================================
# 10. AGE AND SEX
# ============================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=55,
        step=1
    )

    sex_label = st.selectbox(
        "Sex",
        [
            "Female",
            "Male"
        ]
    )

    sex = (
        0
        if sex_label == "Female"
        else 1
    )


# ============================================================
# 11. CHEST PAIN
# ============================================================

with col2:

    cp_label = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )

    cp_mapping = {

        "Typical Angina": 1,

        "Atypical Angina": 2,

        "Non-anginal Pain": 3,

        "Asymptomatic": 4
    }

    cp = cp_mapping[cp_label]


# ============================================================
# 12. BLOOD PRESSURE
# ============================================================

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=140,
        step=1
    )


# ============================================================
# 13. CHOLESTEROL AND HEART RATE
# ============================================================

with col3:

    chol = st.number_input(
        "Serum Cholesterol",
        min_value=50,
        max_value=700,
        value=240,
        step=1
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )


st.divider()


# ============================================================
# 14. ADDITIONAL PARAMETERS
# ============================================================

st.header("⚙️ Additional Parameters")

col1, col2, col3 = st.columns(3)


# ============================================================
# 15. FASTING BLOOD SUGAR
# ============================================================

with col1:

    fbs_label = st.selectbox(
        "Fasting Blood Sugar",
        [
            "≤ 120 mg/dl",
            "> 120 mg/dl"
        ]
    )

    fbs = (
        0
        if fbs_label == "≤ 120 mg/dl"
        else 1
    )


# ============================================================
# 16. RESTING ECG
# ============================================================

with col2:

    restecg_label = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )

    restecg_mapping = {

        "Normal": 0,

        "ST-T Wave Abnormality": 1,

        "Left Ventricular Hypertrophy": 2
    }

    restecg = restecg_mapping[
        restecg_label
    ]


# ============================================================
# 17. EXERCISE ANGINA
# ============================================================

with col3:

    exang_label = st.selectbox(
        "Exercise-Induced Angina",
        [
            "No",
            "Yes"
        ]
    )

    exang = (
        0
        if exang_label == "No"
        else 1
    )


# ============================================================
# 18. ST DEPRESSION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    oldpeak = st.number_input(
        "ST Depression",
        min_value=0.0,
        max_value=10.0,
        value=1.2,
        step=0.1
    )


# ============================================================
# 19. ST SEGMENT SLOPE
# ============================================================

with col2:

    slope_label = st.selectbox(
        "ST Segment Slope",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )

    slope_mapping = {

        "Upsloping": 1,

        "Flat": 2,

        "Downsloping": 3
    }

    slope = slope_mapping[
        slope_label
    ]


# ============================================================
# 20. MAJOR VESSELS
# ============================================================

with col3:

    ca = st.number_input(
        "Number of Major Vessels",
        min_value=0,
        max_value=3,
        value=0,
        step=1
    )


# ============================================================
# 21. THALASSEMIA
# ============================================================

thal_label = st.selectbox(
    "Thalassemia Measurement",
    [
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ]
)

thal_mapping = {

    "Normal": 3,

    "Fixed Defect": 6,

    "Reversible Defect": 7
}

thal = thal_mapping[
    thal_label
]


st.divider()


# ============================================================
# 22. CREATE INPUT DATAFRAME
# ============================================================

patient_data = {

    "age": age,

    "sex": sex,

    "cp": cp,

    "trestbps": trestbps,

    "chol": chol,

    "fbs": fbs,

    "restecg": restecg,

    "thalach": thalach,

    "exang": exang,

    "oldpeak": oldpeak,

    "slope": slope,

    "ca": ca,

    "thal": thal
}


input_df = pd.DataFrame(
    [patient_data],
    columns=FEATURE_NAMES
)


# ============================================================
# 23. ANALYZE BUTTON
# ============================================================

st.header("🔍 Risk Analysis")

analyze = st.button(
    "🔎 Analyze Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# 24. RUN PREDICTION
# ============================================================

if analyze:

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_df
    )[0]

    probability = model.predict_proba(
        input_df
    )[0][1]


    # --------------------------------------------------------
    # Result heading
    # --------------------------------------------------------

    st.divider()

    st.header("📊 Model Result")

    result_col1, result_col2 = st.columns(2)


    # ========================================================
    # 25. RISK RESULT
    # ========================================================

    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ Higher Risk"
            )

        else:

            st.success(
                "✅ Lower Risk"
            )


    # ========================================================
    # 26. PROBABILITY
    # ========================================================

    with result_col2:

        st.metric(
            "Estimated Positive-Class Probability",
            f"{probability * 100:.2f}%"
        )


    st.info(
        """
        This percentage is the model's estimated probability
        for the positive class in the dataset. It is not an
        individual's actual medical probability or diagnosis.
        """
    )


    # ========================================================
    # 27. SHAP EXPLAINABILITY
    # ========================================================

    st.divider()

    st.header(
        "🧠 Why Did the Model Make This Prediction?"
    )

    st.write(
        """
        SHAP shows how individual input features contributed
        to this specific model prediction.
        """
    )


    # --------------------------------------------------------
    # Extract Random Forest
    # --------------------------------------------------------

    rf_model = model.named_steps["model"]


    # --------------------------------------------------------
    # Extract preprocessing step
    # --------------------------------------------------------

    imputer = model.named_steps["imputer"]


    # --------------------------------------------------------
    # Apply the same imputation used during training
    # --------------------------------------------------------

    processed_input = imputer.transform(
        input_df
    )


    # --------------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        rf_model
    )


    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        processed_input
    )


    # --------------------------------------------------------
    # Handle SHAP output formats
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        shap_contributions = (
            shap_values[1][0]
        )

    else:

        if len(shap_values.shape) == 3:

            shap_contributions = (
                shap_values[0, :, 1]
            )

        else:

            shap_contributions = (
                shap_values[0]
            )


    # ========================================================
    # 28. SHAP DATAFRAME
    # ========================================================

    explanation_df = pd.DataFrame({

        "Feature": [
            FEATURE_LABELS[
                feature
            ]
            for feature in FEATURE_NAMES
        ],

        "Value": [
            input_df.iloc[0][feature]
            for feature in FEATURE_NAMES
        ],

        "SHAP Contribution":
            shap_contributions

    })


    explanation_df[
        "Absolute Impact"
    ] = explanation_df[
        "SHAP Contribution"
    ].abs()


    # Sort by strongest influence
    explanation_df = (
        explanation_df
        .sort_values(
            "Absolute Impact",
            ascending=False
        )
    )


    # ========================================================
    # 29. TOP FEATURES TABLE
    # ========================================================

    st.subheader(
        "Top Features Influencing This Prediction"
    )

    top_features = (
        explanation_df
        .head(5)
        .copy()
    )


    st.dataframe(
        top_features[
            [
                "Feature",
                "Value",
                "SHAP Contribution"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # 30. SHAP BAR CHART
    # ========================================================

    st.subheader(
        "Feature Contributions"
    )

    chart_df = (
        top_features
        .sort_values(
            "SHAP Contribution"
        )
    )


    fig, ax = plt.subplots(
        figsize=(9, 5)
    )


    ax.barh(
        chart_df["Feature"],
        chart_df["SHAP Contribution"]
    )


    ax.axvline(
        0,
        linewidth=1
    )


    ax.set_xlabel(
        "SHAP Contribution"
    )


    ax.set_title(
        "Local Feature Contributions"
    )


    plt.tight_layout()


    st.pyplot(
        fig
    )


    plt.close(
        fig
    )


    # ========================================================
    # 31. SIMPLE INTERPRETATION
    # ========================================================

    st.subheader(
        "📌 Model Interpretation"
    )


    positive_features = (
        explanation_df[
            explanation_df[
                "SHAP Contribution"
            ] > 0
        ]
        .head(3)
    )


    negative_features = (
        explanation_df[
            explanation_df[
                "SHAP Contribution"
            ] < 0
        ]
        .head(3)
    )


    # --------------------------------------------------------
    # Positive contributors
    # --------------------------------------------------------

    if not positive_features.empty:

        st.write(
            "**Features pushing the prediction "
            "toward the positive class:**"
        )

        for _, row in (
            positive_features.iterrows()
        ):

            st.write(
                f"• **{row['Feature']}** — "
                f"SHAP: "
                f"{row['SHAP Contribution']:.4f}"
            )


    # --------------------------------------------------------
    # Negative contributors
    # --------------------------------------------------------

    if not negative_features.empty:

        st.write(
            "**Features pushing the prediction "
            "toward the negative class:**"
        )

        for _, row in (
            negative_features.iterrows()
        ):

            st.write(
                f"• **{row['Feature']}** — "
                f"SHAP: "
                f"{row['SHAP Contribution']:.4f}"
            )


    # ========================================================
    # 32. DISCLAIMER
    # ========================================================

    st.divider()

    st.warning(
        """
        ⚠️ Important:

        HealthRisk AI is an educational machine-learning
        project created for demonstrating data preprocessing,
        classification, model tuning and explainable AI.

        It is NOT a medical diagnostic system.

        SHAP values explain model behavior and do not indicate
        medical causation.
        """
    )
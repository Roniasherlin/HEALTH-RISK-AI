import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed.cleveland.data"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading UCI Heart Disease dataset...")

columns = [
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
    "thal",
    "target"
]

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=columns,
    na_values="?"
)

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")


# ============================================================
# 3. BASIC DATA CLEANING
# ============================================================

print("\nCleaning data...")

# Convert all columns to numeric where possible
for column in columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. CREATE BINARY TARGET
# ============================================================

# Original UCI target:
# 0 = No heart disease
# 1-4 = Presence of heart disease

df["target"] = (df["target"] > 0).astype(int)

print("\nTarget distribution:")
print(df["target"].value_counts())


# ============================================================
# 5. SPLIT FEATURES AND TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures:")
print(list(X.columns))


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. LOGISTIC REGRESSION PIPELINE
# ============================================================

logistic_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])


# ============================================================
# 8. RANDOM FOREST PIPELINE
# ============================================================

rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        random_state=42
    ))
])


# ============================================================
# 9. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# 10. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

logistic_pipeline.fit(X_train, y_train)

logistic_metrics = evaluate_model(
    logistic_pipeline,
    X_test,
    y_test,
    "Logistic Regression"
)


# ============================================================
# 11. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf_pipeline.fit(X_train, y_train)

rf_metrics = evaluate_model(
    rf_pipeline,
    X_test,
    y_test,
    "Random Forest"
)


# ============================================================
# 12. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

print("\nStarting Random Forest hyperparameter tuning...")

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("\nBest Random Forest parameters:")
print(grid_search.best_params_)


# ============================================================
# 13. EVALUATE TUNED MODEL
# ============================================================

tuned_rf_metrics = evaluate_model(
    best_rf,
    X_test,
    y_test,
    "Tuned Random Forest"
)


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

rf_model = best_rf.named_steps["model"]

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)


# ============================================================
# 15. SAVE BEST MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "heart_risk_model.pkl"
)

feature_columns_path = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)

metrics_path = os.path.join(
    MODEL_DIR,
    "model_metrics.pkl"
)

importance_path = os.path.join(
    MODEL_DIR,
    "feature_importance.pkl"
)


joblib.dump(best_rf, model_path)

joblib.dump(
    list(X.columns),
    feature_columns_path
)

joblib.dump(
    {
        "logistic_regression": logistic_metrics,
        "random_forest": rf_metrics,
        "tuned_random_forest": tuned_rf_metrics
    },
    metrics_path
)

joblib.dump(
    feature_importance,
    importance_path
)


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"\nModel saved to:")
print(model_path)

print("\nFeature columns saved to:")
print(feature_columns_path)

print("\nMetrics saved to:")
print(metrics_path)

print("\nFeature importance saved to:")
print(importance_path)
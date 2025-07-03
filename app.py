# streamlit_lvad_explainer_app.py

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    url = "heart_failure_clinical_records_dataset.csv"
    df = pd.read_csv(url)
    return df

df = load_data()
st.title("LVAD Candidacy Predictor & SHAP Explainer")

# Input patient index for explanation
patient_index = st.slider("Select a patient from test data (for explanation):", 0, 49, 3)

# Split features and target
X = df.drop(columns=["DEATH_EVENT"])
y = df["DEATH_EVENT"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

# Train & evaluate models
results = []
feature_importance_dict = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    results.append((name, acc, roc))
    if hasattr(model, "feature_importances_"):
        feature_importance_dict[name] = model.feature_importances_
    elif name == "Logistic Regression":
        feature_importance_dict[name] = abs(model.coef_[0])

# Show model performance
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "ROC AUC"]).sort_values(by="ROC AUC", ascending=False)
st.subheader("Model Performance")
st.dataframe(results_df)

# Identify best model
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
st.success(f"Best Model: {best_model_name}")

# SHAP explanation
st.subheader("SHAP Summary Plot")
if best_model_name in ["XGBoost", "Random Forest", "Gradient Boosting"]:
    explainer = shap.TreeExplainer(best_model)
else:
    explainer = shap.Explainer(best_model.predict, X_test)

shap_values = explainer(X_test)
fig_summary = shap.plots.beeswarm(shap_values, max_display=10, show=False)
st.pyplot(bbox_inches='tight', pad_inches=0)

# Explain selected patient
st.subheader(f"Patient {patient_index} Risk Explanation")
patient_data = X_test.iloc[patient_index:patient_index+1]
pred_prob = best_model.predict_proba(patient_data)[0][1] if hasattr(best_model, 'predict_proba') else None
pred_label = best_model.predict(patient_data)[0]

st.write(f"Predicted Risk of Death: {pred_prob*100:.2f}%")
if pred_prob > 0.7:
    st.error("⚠️ High risk detected. Recommend LVAD Evaluation.")
elif pred_prob > 0.4:
    st.warning("Moderate risk. Monitor closely and consider advanced care evaluation.")
else:
    st.success("Low risk. Conservative treatment likely sufficient.")

# Display waterfall plot
st.subheader("Top Contributing Features")
fig_waterfall = shap.plots.waterfall(shap_values[patient_index], max_display=10, show=False)
st.pyplot(bbox_inches='tight', pad_inches=0)

# Show top 5 reasons
st.subheader("AI Explanation Summary")
top_feats = np.argsort(-np.abs(shap_values[patient_index].values))[:5]
for i in top_feats:
    fname = X.columns[i]
    fval = patient_data.iloc[0][i]
    impact = shap_values[patient_index].values[i]
    direction = "increased" if impact > 0 else "reduced"
    st.markdown(f"- **{fname}**: value {fval:.2f} → **{direction}** predicted death risk")

st.caption("Model and explanation are experimental. Clinical decisions should always involve expert judgment.")

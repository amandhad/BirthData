import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import plotly.express as px

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

# Drop useless columns
df = df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop('Attrition', axis=1)
y = df['Attrition']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------
# Model
# ----------------------------
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_estimators=200
)
model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred, output_dict=False)
cm = confusion_matrix(y_test, y_pred)

# ----------------------------
# UI
# ----------------------------
st.title("📊 Employee Attrition Prediction Dashboard")
st.write("Predict whether an employee is at risk of leaving based on their profile.")

# Model performance
st.subheader("🎯 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{acc:.1%}")
col2.metric("ROC-AUC", f"{auc:.3f}")

with st.expander("Detailed metrics"):
    st.text("Classification Report:")
    st.text(report)
    st.text("Confusion Matrix (rows = actual, cols = predicted):")
    st.write(pd.DataFrame(
        cm,
        index=["Actual Stay", "Actual Leave"],
        columns=["Pred Stay", "Pred Leave"]
    ))

# ----------------------------
# EDA Section
# ----------------------------
st.subheader("📈 Attrition Insights")

fig1 = px.histogram(df, x="Attrition", title="Attrition Distribution")
st.plotly_chart(fig1)

# ----------------------------
# USER INPUT
# ----------------------------
st.subheader("🧑‍💼 Enter Employee Information")

age = st.slider("Age", 18, 60, 30)
income = st.slider("Monthly Income", 1000, 20000, 5000)
overtime = st.selectbox("Overtime", ["No", "Yes"])
job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
years = st.slider("Years at Company", 0, 40, 5)

# Convert input
input_data = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [income],
    'OverTime': [1 if overtime == "Yes" else 0],
    'JobSatisfaction': [job_sat],
    'YearsAtCompany': [years]
})

# Align columns
input_data = input_data.reindex(columns=X.columns, fill_value=0)

input_scaled = scaler.transform(input_data)

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict Attrition Risk"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Attrition ({prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Attrition ({prob:.2f})")

    st.subheader("📊 Feature Importance")

    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by="Importance", ascending=False).head(10)

    fig2 = px.bar(feat_df, x="Importance", y="Feature", orientation='h')
    st.plotly_chart(fig2)

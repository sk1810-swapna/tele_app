import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page setup
st.set_page_config(page_title="Churn Prediction Dashboard", layout="centered")
st.title("📉 Churn Prediction Dashboard")

# Sidebar inputs
st.sidebar.header("🔧 Customer Feature Input")
total_day_minutes = st.sidebar.slider("Total Day Minutes", 0, 350, 180)
customer_service_calls = st.sidebar.slider("Customer Service Calls", 0, 10, 1)
international_plan = st.sidebar.selectbox("International Plan", ["No", "Yes"])
voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan", ["No", "Yes"])

# Convert categorical to binary
intl_plan_bin = 1 if international_plan == "Yes" else 0
vm_plan_bin = 1 if voice_mail_plan == "Yes" else 0

# Create input DataFrame
input_data = pd.DataFrame([{
    "total_day_minutes": total_day_minutes,
    "customer_service_calls": customer_service_calls,
    "international_plan": intl_plan_bin,
    "voice_mail_plan": vm_plan_bin
}])

# Simulated full dataset
X_full = pd.DataFrame([
    {"total_day_minutes": 100, "customer_service_calls": 1, "international_plan": 0, "voice_mail_plan": 1},
    {"total_day_minutes": 250, "customer_service_calls": 5, "international_plan": 1, "voice_mail_plan": 0},
    {"total_day_minutes": 180, "customer_service_calls": 2, "international_plan": 0, "voice_mail_plan": 1},
    {"total_day_minutes": 300, "customer_service_calls": 7, "international_plan": 1, "voice_mail_plan": 0},
    {"total_day_minutes": 120, "customer_service_calls": 0, "international_plan": 0, "voice_mail_plan": 1},
    {"total_day_minutes": 200, "customer_service_calls": 3, "international_plan": 1, "voice_mail_plan": 0},
    {"total_day_minutes": 90, "customer_service_calls": 0, "international_plan": 0, "voice_mail_plan": 1},
    {"total_day_minutes": 220, "customer_service_calls": 4, "international_plan": 1, "voice_mail_plan": 0},
    {"total_day_minutes": 160, "customer_service_calls": 2, "international_plan": 0, "voice_mail_plan": 1},
    {"total_day_minutes": 280, "customer_service_calls": 6, "international_plan": 1, "voice_mail_plan": 0},
])
y_full = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Stay, 1 = Churn

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=42)

# Define models
models = {
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train all models and store test accuracy
model_summaries = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    churn_rate = sum(y_pred) / len(y_pred)
    model_summaries[name] = {
        "model": model,
        "accuracy": acc,
        "churn_rate": churn_rate
    }

# Identify best model based on test accuracy
best_model_name = max(model_summaries, key=lambda x: model_summaries[x]["accuracy"])
best_model = model_summaries[best_model_name]["model"]
best_accuracy = model_summaries[best_model_name]["accuracy"]
best_churn_rate = model_summaries[best_model_name]["churn_rate"]

# Dropdown to select model
st.subheader("🔽 Choose a Model")
selected_model_name = st.selectbox("Select Model", list(models.keys()))
selected_model = model_summaries[selected_model_name]["model"]
selected_accuracy = model_summaries[selected_model_name]["accuracy"]
selected_churn_rate = model_summaries[selected_model_name]["churn_rate"]

# Predict
prediction = selected_model.predict(input_data)[0]

# Display result
st.subheader("📊 Prediction Result")
if prediction == 1:
    st.error("⚠️ This customer is likely to CHURN.")
else:
    st.success("✅ This customer is likely to STAY loyal.")

# Visualization
fig, ax = plt.subplots(figsize=(4, 3))
sns.barplot(x=["Stay", "Churn"], y=[1 - prediction, prediction], palette="Set2", ax=ax)
ax.set_title("Churn Prediction Breakdown")
ax.set_ylabel("Probability (simulated)")
st.pyplot(fig)

# Model summary
st.subheader("📌 Model Summary")

summary_text = {
    "SVM": "Support Vector Machine is ideal for binary classification and finds optimal boundaries between churn and loyalty. Best for clean, well-separated data.",
    "KNN": "K-Nearest Neighbors predicts churn based on similarity to other customers. Simple and effective for small datasets.",
    "Decision Tree": "Decision Trees split data based on feature thresholds. They're interpretable and good for rule-based churn detection.",
    "Random Forest": "Random Forest combines multiple decision trees for robust predictions. It handles feature interactions well and is highly accurate even with noisy data."
}

st.markdown(f"**Model Selected:** `{selected_model_name}`")
st.markdown(f"**Test Accuracy:** `{selected_accuracy:.2%}`")
st.markdown(f"**Churn Prediction Rate (Test Set):** `{selected_churn_rate:.2%}`")
st.markdown(f"**Use Case:** {summary_text[selected_model_name]}")

# Highlight best model
st.subheader("🏆 Best Performing Model")
st.markdown(f"**Best Model Based on Test Accuracy:** `{best_model_name}`")
st.markdown(f"**Accuracy:** `{best_accuracy:.2%}`")
st.markdown(f"**Churn Prediction Rate:** `{best_churn_rate:.2%}`")
st.success(f"{best_model_name} is currently the most accurate model for predicting churn in this setup.")

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

# Simulated training data (for demo purposes)
# Normally you'd load and train on real data
X_train = pd.DataFrame([
    {"total_day_minutes": 100, "customer_service_calls": 1, "international_plan": 0, "voice_mail_plan": 1},
    {"total_day_minutes": 250, "customer_service_calls": 5, "international_plan": 1, "voice_mail_plan": 0},
    {"total_day_minutes": 180, "customer_service_calls": 2, "international_plan": 0, "voice_mail_plan": 1},
    {"total_day_minutes": 300, "customer_service_calls": 7, "international_plan": 1, "voice_mail_plan": 0},
    {"total_day_minutes": 120, "customer_service_calls": 0, "international_plan": 0, "voice_mail_plan": 1},
])
y_train = [0, 1, 0, 1, 0]  # 0 = Stay, 1 = Churn

# Define models
models = {
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Dropdown to select model
st.subheader("🔽 Choose a Model")
selected_model_name = st.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

# Train and predict
selected_model.fit(X_train, y_train)
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
    "SVM": "Support Vector Machine is ideal for binary classification and finds optimal boundaries between churn and loyalty.",
    "KNN": "K-Nearest Neighbors predicts churn based on similarity to other customers. Simple and intuitive.",
    "Decision Tree": "Decision Trees split data based on feature thresholds. Great for rule-based churn detection.",
    "Random Forest": "Random Forest combines multiple decision trees for robust predictions. Highly accurate and handles feature interactions well."
}
st.markdown(f"**Model:** {selected_model_name}")
st.markdown(f"**Summary:** {summary_text[selected_model_name]}")

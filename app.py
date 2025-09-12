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

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("telecommunications_churn(1).csv")
    return df

df = load_data()

# Feature selection
features = ["total_day_minutes", "customer_service_calls", "international_plan", "voice_mail_plan"]
X = df[features]
y = df["churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train all models and store accuracy
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

# Sidebar for user input
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

# Algorithm summary
st.subheader("📌 Model Summary")
summary_text = {
    "SVM": "Support Vector Machine is effective for high-dimensional spaces and binary classification. It finds the optimal boundary between churn and loyalty.",
    "KNN": "K-Nearest Neighbors predicts churn based on similarity to other customers. It's simple and works well with well-separated classes.",
    "Decision Tree": "Decision Trees split data based on feature thresholds. They're interpretable and good for rule-based churn detection.",
    "Random Forest": "Random Forest combines multiple decision trees for robust predictions. It's highly accurate and handles feature interactions well."
}

st.markdown(f"**Model:** {selected_model_name}")
st.markdown(f"**Accuracy:** `{selected_accuracy:.4f}`")
st.markdown(f"**Predicted Churn Rate (on test set):** `{selected_churn_rate:.2%}``")
st.markdown(f"**Summary:** {summary_text[selected_model_name]}")

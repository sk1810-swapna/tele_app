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
st.set_page_config(page_title="📉 Churn Prediction Dashboard", layout="centered")
st.title("📉 Churn Prediction Dashboard")

# ✅ Sample dataset
df = pd.DataFrame([
    {"total_day_minutes": 100, "customer_service_calls": 1, "international_plan": 0, "voice_mail_plan": 1, "churn": 0},
    {"total_day_minutes": 250, "customer_service_calls": 5, "international_plan": 1, "voice_mail_plan": 0, "churn": 1},
    {"total_day_minutes": 180, "customer_service_calls": 2, "international_plan": 0, "voice_mail_plan": 1, "churn": 0},
    {"total_day_minutes": 300, "customer_service_calls": 7, "international_plan": 1, "voice_mail_plan": 0, "churn": 1},
    {"total_day_minutes": 120, "customer_service_calls": 0, "international_plan": 0, "voice_mail_plan": 1, "churn": 0},
    {"total_day_minutes": 200, "customer_service_calls": 3, "international_plan": 1, "voice_mail_plan": 0, "churn": 1},
    {"total_day_minutes": 90,  "customer_service_calls": 0, "international_plan": 0, "voice_mail_plan": 1, "churn": 0},
    {"total_day_minutes": 220, "customer_service_calls": 4, "international_plan": 1, "voice_mail_plan": 0, "churn": 1},
    {"total_day_minutes": 160, "customer_service_calls": 2, "international_plan": 0, "voice_mail_plan": 1, "churn": 0},
    {"total_day_minutes": 280, "customer_service_calls": 6, "international_plan": 1, "voice_mail_plan": 0, "churn": 1},
])

# Prepare features and target
X = df.drop("churn", axis=1)
y = df["churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define models
models = {
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Model summaries
summary_text = {
    "SVM": "Support Vector Machine finds optimal boundaries between churn and loyalty. Best for clean, well-separated data.",
    "KNN": "K-Nearest Neighbors predicts churn based on similarity to other customers. Simple and effective for small datasets.",
    "Decision Tree": "Decision Trees split data based on feature thresholds. They're interpretable and good for rule-based churn detection.",
    "Random Forest": "Random Forest combines multiple decision trees for robust predictions. It handles feature interactions well and is highly accurate even with noisy data."
}

# Train and evaluate models
model_scores = {}
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_scores[name] = acc
    trained_models[name] = model

# ✅ Tie-breaking logic: prefer Random Forest if tied
sorted_models = sorted(
    model_scores.items(),
    key=lambda x: (-x[1], x[0] != "Random Forest")  # Highest accuracy, then prefer RF
)
best_model_name = sorted_models[0][0]
best_model = trained_models[best_model_name]

# Sidebar inputs
st.sidebar.header("🔧 Customer Feature Input")

# total_day_minutes slider
total_day_minutes = st.sidebar.slider("Total Day Minutes", int(X["total_day_minutes"].min()), int(X["total_day_minutes"].max()), int(X["total_day_minutes"].mean()))

# customer_service_calls slider (max 9)
customer_service_calls = st.sidebar.slider("Customer Service Calls", 0, 9, int(X["customer_service_calls"].mean()))

# international_plan dropdown
international_plan = st.sidebar.selectbox("International Plan", ["No", "Yes"])
intl_plan_bin = 1 if international_plan == "Yes" else 0

# voice_mail_plan dropdown
voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan", ["No", "Yes"])
vm_plan_bin = 1 if voice_mail_plan == "Yes" else 0

# Create input DataFrame
input_df = pd.DataFrame([{
    "total_day_minutes": total_day_minutes,
    "customer_service_calls": customer_service_calls,
    "international_plan": intl_plan_bin,
    "voice_mail_plan": vm_plan_bin
}])

# Dropdown to select model
st.subheader("🔽 Choose a Model")
selected_model_name = st.selectbox("Select Model", list(models.keys()), index=list(models.keys()).index(best_model_name))
selected_model = trained_models[selected_model_name]
prediction = selected_model.predict(input_df)[0]

# 🎯 Churn Prediction Visualization
st.subheader("📊 Churn Prediction Result")
if prediction == 1:
    st.error("⚠️ This customer is likely to CHURN.")
else:
    st.success("✅ This customer is likely to STAY loyal.")

fig, ax = plt.subplots(figsize=(4, 3))
sns.barplot(x=["Stay", "Churn"], y=[1 - prediction, prediction], palette="Set2", ax=ax)
ax.set_title("Churn Prediction Breakdown")
ax.set_ylabel("Probability (simulated)")
st.pyplot(fig)

# 📌 Model Summary
st.subheader("📌 Model Summary")
st.markdown(f"**Model Selected:** `{selected_model_name}`")
st.markdown(f"**Test Accuracy:** `{model_scores[selected_model_name]:.4f}`")
st.markdown(f"**Use Case:** {summary_text[selected_model_name]}")

# 🏆 Best Model Display
st.subheader("🏆 Best Performing Model")
st.markdown(f"**Best Model Based on Test Accuracy:** `{best_model_name}`")
st.markdown(f"**Accuracy:** `{model_scores[best_model_name]:.4f}`")
st.success(f"{best_model_name} is currently the most accurate model for predicting churn in this setup.")

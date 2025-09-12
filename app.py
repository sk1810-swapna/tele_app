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
from sklearn.metrics import accuracy_score, classification_report

# Page setup
st.set_page_config(page_title="Churn Prediction Dashboard", layout="centered")
st.title("📉 Churn Prediction Dashboard")

# ✅ Sample dataset embedded directly
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
X = df.drop('churn', axis=1)
y = df['churn']

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

# Evaluate models
accuracy_results = {}
reports = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = acc
    reports[name] = classification_report(y_test, y_pred, output_dict=True)

# Sort and identify best model
results_df = pd.DataFrame.from_dict(accuracy_results, orient='index', columns=['Accuracy'])
results_df = results_df.sort_values(by='Accuracy', ascending=False)
best_model_name = results_df.idxmax().values[0]
best_accuracy = results_df.max().values[0]
best_model = models[best_model_name]

# Sidebar inputs
st.sidebar.header("🔧 Customer Feature Input")
input_data = {}
for col in X.columns:
    min_val = int(X[col].min())
    max_val = int(X[col].max())
    default_val = int(X[col].mean())
    input_data[col] = st.sidebar.slider(col, min_val, max_val, default_val)
input_df = pd.DataFrame([input_data])

# Model selection
selected_model_name = st.selectbox("🔽 Choose a Model", list(models.keys()))
selected_model = models[selected_model_name]
selected_model.fit(X_train, y_train)
prediction = selected_model.predict(input_df)[0]
selected_accuracy = accuracy_results[selected_model_name]
selected_report = reports[selected_model_name]

# Prediction result
st.subheader("📊 Prediction Result")
if prediction == 1:
    st.error("⚠️ This customer is likely to CHURN.")
else:
    st.success("✅ This customer is likely to STAY loyal.")

# Accuracy chart
st.subheader("📈 Model Accuracy Comparison")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=results_df.index, y=results_df['Accuracy'], palette='viridis', ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")
for bar in ax.patches:
    acc = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.4f}", ha='center', va='bottom', fontsize=9)
st.pyplot(fig)

# Model summary
st.subheader("📌 Model Summary")
st.markdown(f"**Model Selected:** `{selected_model_name}`")
st.markdown(f"**Test Accuracy:** `{selected_accuracy:.4f}`")
st.markdown(f"**Precision (Churn):** `{selected_report['1']['precision']:.2f}`")
st.markdown(f"**Recall (Churn):** `{selected_report['1']['recall']:.2f}`")
st.markdown(f"**F1-Score (Churn):** `{selected_report['1']['f1-score']:.2f}`")

# Best model highlight
st.subheader("🏆 Best Performing Model")
st.markdown(f"**Best Model Based on Test Accuracy:** `{best_model_name}`")
st.markdown(f"**Accuracy:** `{best_accuracy:.4f}`")
st.success(f"{best_model_name} is currently the most accurate model for predicting churn in this setup.")

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
st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("📉 Churn Prediction Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("telecommunications_churn(1).csv", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded!")

    # Check for 'churn' column
    if 'churn' not in df.columns:
        st.error("Dataset must contain a 'churn' column.")
    else:
        # Prepare features and target
        X = df.drop('churn', axis=1).select_dtypes(include=['number'])
        y = df['churn']
        y = y.loc[X.index]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Define models
        model_options = {
            "SVM": make_pipeline(StandardScaler(), SVC()),
            "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        # Dropdown to select model
        st.subheader("🔽 Choose a Model for Churn Prediction")
        selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
        selected_model = model_options[selected_model_name]

        # Train and predict
        selected_model.fit(X_train, y_train)
        y_pred = selected_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Display accuracy
        st.markdown(f"### ✅ {selected_model_name} Accuracy: `{acc:.4f}`")

        # Visualize churn prediction
        st.subheader("📊 Predicted Churn Distribution")
        viz_df = pd.DataFrame({
            "Predicted Churn": y_pred
        })

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Predicted Churn", data=viz_df, palette="Set2", ax=ax)
        ax.set_title(f"Churn Prediction - {selected_model_name}")
        ax.set_xlabel("Predicted Churn (0 = Stay, 1 = Churn)")
        ax.set_ylabel("Customer Count")
        st.pyplot(fig)


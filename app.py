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

# Page config
st.set_page_config(page_title="Churn Model Explorer", layout="wide")

st.title("📊 Churn Prediction Model Explorer")

# Upload CSV
uploaded_file = st.file_uploader("Upload your churn dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")

    # Prepare features and target
    if 'churn' not in df.columns:
        st.error("Dataset must contain a 'churn' column.")
    else:
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
        st.subheader("🤖 Select and Evaluate a Model")
        selected_model_name = st.selectbox("Choose a model:", list(model_options.keys()))
        selected_model = model_options[selected_model_name]

        # Train and evaluate selected model
        selected_model.fit(X_train, y_train)
        y_pred = selected_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.markdown(f"### 📈 {selected_model_name} Performance")
        st.write(f"**Accuracy:** {acc:.4f}")
        st.text(report)

        # Compare all models
        st.subheader("📊 Compare All Models (Accuracy)")
        accuracy_results = {}
        for name, model in model_options.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracy_results[name] = acc

        results_df = pd.DataFrame.from_dict(accuracy_results, orient='index', columns=['Accuracy'])
        results_df = results_df.sort_values(by='Accuracy', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=results_df.index, y=results_df['Accuracy'], palette='viridis', ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        for bar in ax.patches:
            acc = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                acc + 0.01,
                f"{acc:.4f}",
                ha='center',
                va='bottom',
                fontsize=9,
                color='black'
            )
        st.pyplot(fig)

        # Best model
        best_model = results_df.idxmax().values[0]
        best_accuracy = results_df.max().values[0]
        st.success(f"🏆 Best Model: {best_model} with Accuracy = {best_accuracy:.4f}")

        # Churn prediction visualization using best model
        st.subheader("📉 Churn Prediction Distribution (Best Model)")
        final_model = model_options[best_model]
        final_model.fit(X_train, y_train)
        final_preds = final_model.predict(X_test)

        viz_df = pd.DataFrame({
            "Actual Churn": y_test.values,
            "Predicted Churn": final_preds
        })

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Predicted Churn", data=viz_df, palette="Set2", ax=ax)
        ax.set_title(f"Predicted Churn - {best_model}")
        ax.set_xlabel("Predicted Churn (0 = No, 1 = Yes)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)


def main():

    st.title("Binary Classification Web App")

    st.sidebar.title("Binary Classification Web App")

    st.markdown("Are your mushrooms edible or poisonous?")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    @st.cache_data
    def load_data():

        data = pd.read_csv("mushrooms.csv")

        # Encode categorical columns
        for col in data.columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])

        return data

    # ---------------------------------------------------------
    # Split data
    # ---------------------------------------------------------
    @st.cache_data
    def split_data(df):

        y = df["type"]
        x = df.drop(columns=["type"])

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.3,
            random_state=0,
            stratify=y
        )

        return x_train, x_test, y_train, y_test

    # ---------------------------------------------------------
    # Plot metrics
    # ---------------------------------------------------------
    def plot_metrics(metrics_list, model, x_test, y_test):

        # Confusion Matrix
        if "Confusion Matrix" in metrics_list:

            st.subheader("Confusion Matrix")

            y_pred = model.predict(x_test)

            cm = confusion_matrix(y_test, y_pred)

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["edible", "poisonous"]
            )

            fig, ax = plt.subplots()

            disp.plot(
                ax=ax,
                cmap="Blues"
            )

            st.pyplot(fig)

            plt.close(fig)

        # ROC Curve
        if "ROC Curve" in metrics_list:

            st.subheader("ROC Curve")

            y_prob = model.predict_proba(x_test)[:, 1]

            fpr, tpr, _ = roc_curve(
                y_test,
                y_prob
            )

            fig, ax = plt.subplots()

            ax.plot(
                fpr,
                tpr,
                label="ROC Curve"
            )

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--"
            )

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()

            st.pyplot(fig)

            plt.close(fig)

        # Precision-Recall Curve
        if "Precision-Recall Curve" in metrics_list:

            st.subheader("Precision-Recall Curve")

            y_prob = model.predict_proba(x_test)[:, 1]

            precision, recall, _ = precision_recall_curve(
                y_test,
                y_prob
            )

            fig, ax = plt.subplots()

            ax.plot(
                recall,
                precision
            )

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")

            st.pyplot(fig)

            plt.close(fig)

    # ---------------------------------------------------------
    # Load and split data
    # ---------------------------------------------------------
    df = load_data()

    x_train, x_test, y_train, y_test = split_data(df)

    # ---------------------------------------------------------
    # Classifier selection
    # ---------------------------------------------------------
    st.sidebar.subheader("Choose Classifier")

    classifier = st.sidebar.selectbox(
        "Classifier",
        (
            "Support Vector Machine (SVM)",
            "Logistic Regression",
            "Random Forest"
        )
    )

    # ---------------------------------------------------------
    # Metrics selection
    # ---------------------------------------------------------
    metrics = st.sidebar.multiselect(
        "What metrics to plot?",
        (
            "Confusion Matrix",
            "ROC Curve",
            "Precision-Recall Curve"
        )
    )

    # =========================================================
    # SUPPORT VECTOR MACHINE
    # =========================================================
    if classifier == "Support Vector Machine (SVM)":

        st.sidebar.subheader("Model Hyperparameters")

        C = st.sidebar.number_input(
            "C (Regularization parameter)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            key="svm_C"
        )

        kernel = st.sidebar.radio(
            "Kernel",
            ("rbf", "linear"),
            key="svm_kernel"
        )

        gamma = st.sidebar.radio(
            "Gamma (kernel coefficient)",
            ("scale", "auto"),
            key="svm_gamma"
        )

        if st.sidebar.button(
            "Classify",
            key="svm_classify"
        ):

            st.subheader("Support Vector Machine (SVM) Results")

            model = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                probability=True
            )

            model.fit(
                x_train,
                y_train
            )

            y_pred = model.predict(x_test)

            accuracy = model.score(
                x_test,
                y_test
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision, 2))
            st.write("Recall:", round(recall, 2))

            plot_metrics(
                metrics,
                model,
                x_test,
                y_test
            )

    # =========================================================
    # LOGISTIC REGRESSION
    # =========================================================
    elif classifier == "Logistic Regression":

        st.sidebar.subheader("Model Hyperparameters")

        C = st.sidebar.number_input(
            "C (Regularization parameter)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            key="lr_C"
        )

        max_iter = st.sidebar.slider(
            "Maximum number of iterations",
            min_value=100,
            max_value=500,
            value=100,
            key="lr_max_iter"
        )

        if st.sidebar.button(
            "Classify",
            key="lr_classify"
        ):

            st.subheader("Logistic Regression Results")

            model = LogisticRegression(
                C=C,
                max_iter=max_iter
            )

            model.fit(
                x_train,
                y_train
            )

            y_pred = model.predict(x_test)

            accuracy = model.score(
                x_test,
                y_test
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision, 2))
            st.write("Recall:", round(recall, 2))

            plot_metrics(
                metrics,
                model,
                x_test,
                y_test
            )

    # =========================================================
    # RANDOM FOREST
    # =========================================================
    elif classifier == "Random Forest":

        st.sidebar.subheader("Model Hyperparameters")

        n_estimators = st.sidebar.slider(
            "Number of estimators",
            min_value=10,
            max_value=500,
            value=100,
            key="rf_n_estimators"
        )

        max_depth = st.sidebar.slider(
            "Maximum depth",
            min_value=1,
            max_value=20,
            value=10,
            key="rf_max_depth"
        )

        if st.sidebar.button(
            "Classify",
            key="rf_classify"
        ):

            st.subheader("Random Forest Results")

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=0
            )

            model.fit(
                x_train,
                y_train
            )

            y_pred = model.predict(x_test)

            accuracy = model.score(
                x_test,
                y_test
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision, 2))
            st.write("Recall:", round(recall, 2))

            plot_metrics(
                metrics,
                model,
                x_test,
                y_test
            )

    # ---------------------------------------------------------
    # Show raw data
    # ---------------------------------------------------------
    if st.sidebar.checkbox(
        "Show raw data",
        value=False
    ):

        st.subheader("Mushroom Data Set (Classification)")

        st.dataframe(df)


if __name__ == "__main__":
    main()

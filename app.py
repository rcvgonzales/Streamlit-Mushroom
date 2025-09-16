import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import streamlit as st 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay 
from sklearn.metrics import precision_score, recall_score 


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

    @st.cache_data
    def load_data():
        data = pd.read_csv('mushroom_data_all.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache_data
    def split(df):
        y = df['type']
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",
                                      ("Support Vector Machine (SVM)",
                                       "Logistic Regression",
                                       "Random Forest"))

    if st.sidebar.checkbox("Show Raw Data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)

    # --- SVM ---
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify_svm'):
            st.header("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))

            plot_metrics(model, x_test, y_test, metrics, class_names)

    # --- Logistic Regression ---
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify_lr'):
            st.header("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))

            plot_metrics(model, x_test, y_test, metrics, class_names)

    # --- Random Forest ---
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap_choice = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        bootstrap = True if bootstrap_choice == 'True' else False

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify_rf'):
            st.header("Random Forest Results")
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                bootstrap=bootstrap,
                n_jobs=-1
            )
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))

            plot_metrics(model, x_test, y_test, metrics, class_names)


def plot_metrics(model, x_test, y_test, metrics_list, class_names):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names)
        st.pyplot(disp.figure_)

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        disp = RocCurveDisplay.from_estimator(model, x_test, y_test)
        st.pyplot(disp.figure_)

    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        disp = PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
        st.pyplot(disp.figure_)


if __name__ == '__main__':
    main()

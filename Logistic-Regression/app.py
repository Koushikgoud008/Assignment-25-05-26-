import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.eda import show_eda

st.set_page_config(page_title="Iris Classification", layout="wide")

st.title("Iris Species Classification using Logistic Regression")

if not os.path.exists('data/iris.csv'):
    from src.data_preprocessing import load_and_save_data
    df = load_and_save_data()
else:
    df = pd.read_csv('data/iris.csv')

st.subheader("Dataset Preview")
st.markdown(df.head().to_html(classes='table table-striped'), unsafe_allow_html=True)

st.subheader("Dataset Information")
st.write(df.shape)
st.text(str(df.dtypes))

st.subheader("Missing Values")
st.text(str(df.isnull().sum()))

show_eda(df)

st.header("Feature Engineering")

X = df.drop('target', axis=1)
y = df['target']

st.write("Features Shape:", X.shape)
st.write("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write("Training Data Shape:", X_train_scaled.shape)
st.write("Testing Data Shape:", X_test_scaled.shape)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

st.success("Model Training Completed")

y_pred = model.predict(X_test_scaled)

st.header("Model Evaluation")

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy Score: {accuracy:.4f}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/logistic_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

st.success("Model Saved Successfully")

st.header("Predict New Data")

sepal_length = st.number_input('Sepal Length (cm)', value=5.1)
sepal_width = st.number_input('Sepal Width (cm)', value=3.5)
petal_length = st.number_input('Petal Length (cm)', value=1.4)
petal_width = st.number_input('Petal Width (cm)', value=0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Iris Class (0=Setosa, 1=Versicolor, 2=Virginica): {prediction[0]}")
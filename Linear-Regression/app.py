import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
csv_path=os.path.join(BASE_DIR,"data","sp-historical.csv")
model_path=os.path.join(BASE_DIR,"models","linear_regression.pkl")
scaler_path=os.path.join(BASE_DIR,"models","scaler.pkl")

df=pd.read_csv(csv_path)
model=joblib.load(model_path)
scaler=joblib.load(scaler_path)
df["Date"]=pd.to_datetime(df["Date"])
st.set_page_config(page_title="S&P500 Predictor",layout="wide")
st.title("S&P500 Linear Regression Dashboard")
col1,col2,col3=st.columns(3)

with col1:
    st.metric("Latest Close",round(df["Close"].iloc[-1],2))

with col2:
    st.metric("Highest Close",round(df["Close"].max(),2))

with col3:
    st.metric("Average Close",round(df["Close"].mean(),2))

tab1,tab2,tab3=st.tabs(["Dashboard","Prediction","Dataset"])

with tab1:
    st.subheader("Close Price Trend")
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(df["Date"],df["Close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    st.pyplot(fig)
    st.subheader("Open vs Close")
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(df["Open"],df["Close"])
    ax.set_xlabel("Open")
    ax.set_ylabel("Close")
    st.pyplot(fig)
    st.subheader("Close Price Distribution")
    fig,ax=plt.subplots(figsize=(10,5))
    ax.hist(df["Close"],bins=30)
    st.pyplot(fig)

with tab2:
    open_price=st.number_input("Open Price",value=float(df["Open"].iloc[-1]))
    if st.button("Predict"):
        data=np.array([[open_price]])
        data=scaler.transform(data)
        prediction=model.predict(data)
        st.success(f"Predicted Close Price:{prediction[0]:.2f}")
        fig,ax=plt.subplots()
        ax.bar(["Predicted Close"],[prediction[0]])
        st.pyplot(fig)

with tab3:
    st.dataframe(df.tail(50))
    st.write("Rows:",df.shape[0])
    st.write("Columns:",df.shape[1])
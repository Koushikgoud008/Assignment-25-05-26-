# S&P500 Closing Price Prediction
## Overview
This project predicts the S&P500 closing price using historical market data and a Linear Regression model. The application is deployed with Streamlit and includes interactive visualizations and real-time predictions.
## Features
- Data Cleaning and Preprocessing
- Feature Engineering
- Linear Regression Model
- Model Evaluation
- Interactive Streamlit Dashboard
- Price Prediction
- Market Trend Visualizations
## Dataset Features
- Open
- High
- Low
- Close (Target)
- Volume
- Year
- Month
- Day
- Volatility
- MA5 (5-Day Moving Average)
- MA20 (20-Day Moving Average)
## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Streamlit
- Joblib
## Project Structure
```text
linear-regression-hull/
│
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── sp-historical.csv
└── models/
    ├── linear_regression.pkl
    └── scaler.pkl
## Run the Project
Install dependencies:
command prompt:
pip install -r requirements.txt

Train the model:
command prompt:
python train_model.py

Run the Streamlit app:
command prompt:
streamlit run app.py

## Dashboard
- Closing Price Trend
- Volume Trend
- Moving Average Analysis
- Price Distribution
- Dataset Preview
- Closing Price Prediction
## Model
Algorithm: Linear Regression
Evaluation Metrics:
- R² Score
- MAE
- RMSE
## Future Improvements
- Random Forest
- XGBoost
- LSTM Forecasting
- Live Market Data Integration
## Author
Varun Kumar
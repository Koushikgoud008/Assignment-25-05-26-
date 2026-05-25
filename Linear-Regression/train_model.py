import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
csv_path=os.path.join(BASE_DIR,"data","sp-historical.csv")
model_path=os.path.join(BASE_DIR,"models","linear_regression.pkl")
scaler_path=os.path.join(BASE_DIR,"models","scaler.pkl")
os.makedirs(os.path.join(BASE_DIR,"models"),exist_ok=True)
df=pd.read_csv(csv_path)

X=df[["Open"]]
y=df["Close"]
split=int(len(df)*0.8)
X_train=X[:split]
X_test=X[split:]
y_train=y[:split]
y_test=y[split:]

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=LinearRegression()
model.fit(X_train,y_train)
pred=model.predict(X_test)

print("R2:",r2_score(y_test,pred))
print("MAE:",mean_absolute_error(y_test,pred))
print("RMSE:",mean_squared_error(y_test,pred)**0.5)

joblib.dump(model,model_path)
joblib.dump(scaler,scaler_path)


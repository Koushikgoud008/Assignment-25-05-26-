from sklearn.linear_model import LogisticRegression
import joblib
import os

def train(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/logistic_model.pkl')
    return model
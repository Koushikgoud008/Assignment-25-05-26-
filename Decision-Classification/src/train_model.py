from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def train(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/dt_clf_model.pkl')
    return model
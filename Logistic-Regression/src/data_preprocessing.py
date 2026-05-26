import pandas as pd
from sklearn.datasets import load_iris
import os

def load_and_save_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/iris.csv', index=False)
    return df
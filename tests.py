import pandas as pd
import pytest
from ml.data import process_data


@pytest.fixture()
def data():
    return pd.read_csv('data/census.csv')

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def test_preprocess_data(data):
    X_train, y_train, encoder, lb = process_data(data, cat_features, label='salary', training=True)
    assert all(y == 0 or y == 1 for y in y_train)
    assert X_train.shape[0] == data.shape[0]

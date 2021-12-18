import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


@pytest.fixture()
def data():
    return pd.read_csv('data/census.csv')


@pytest.fixture()
def model():
    X_train = [[0, 0]]
    y_train = [1]
    return train_model(X_train, y_train)


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


def test_train_model():
    X_train = [[0, 0]]
    y_train = [1]
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    y = [1, 0, 1]
    preds = [1, 0, 1]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1
    assert recall == 1
    assert fbeta == 1


def test_inference(model):
    X = [[0, 0]]
    predictions = inference(model, X)
    assert predictions == [1]


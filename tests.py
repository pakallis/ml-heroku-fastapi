from fastapi.testclient import TestClient
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from main import app


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


def test_train_and_inference():
    X_train = [[0, 0]]
    y_train = [1]
    model = train_model(X_train, y_train)

    predictions = inference(model, X_train)
    assert predictions == [1]


client = TestClient(app)


def test_client_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'message': 'Salary predictions API'}


def test_client_post_inference_lt_50k():
    data = {
        'age': 39,
        'workclass': 'State-gov',
        'fnlgt': 77516,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
    }
    response = client.post("/inference/", json=data)
    assert response.status_code == 200
    assert response.json() == {'salary': '<=50K'}


def test_client_post_inference_gt_50k():
    data = {
        'age': 52,
        'workclass': 'Self-emp-inc',
        'fnlgt': 287927,
        'education': 'HS-grad',
        'education_num': 9,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Wife',
        'race': 'White',
        'sex': 'Female',
        'capital_gain': 15024,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States',
    }
    response = client.post("/inference/", json=data)
    assert response.status_code == 200
    assert response.json() == {'salary': '>50K'}

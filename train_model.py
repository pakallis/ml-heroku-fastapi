# Script to train machine learning model.

from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model


# Add the necessary imports for the starter code.
data = pd.read_csv("data/census.csv")

# Add code to load in the data.

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True, encoder=encoder, lb=lb
)


model = train_model(X_train, y_train)
# Train and save a model.
dump(model, 'model/random_forest.joblib')
dump(encoder, 'model/encoder.joblib')

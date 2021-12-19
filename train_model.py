# Script to train machine learning model.

import logging
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from ml.const import cat_features


logging.basicConfig(level=logging.INFO)



def metrics(data, education, lb):
    data = data[data.education == education]
    model = load('model/random_forest.joblib')
    encoder = load('model/encoder.joblib')
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, X)
    return compute_model_metrics(y, preds)


# Add the necessary imports for the starter code.
data = pd.read_csv("data/census.csv")

# Add code to load in the data.

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


education_set = set(data['education'])


model = train_model(X_train, y_train)

# Train and save a model.
dump(model, 'model/random_forest.joblib')
dump(encoder, 'model/encoder.joblib')

# Calculate metrics in test set

logger = logging.getLogger()
precision, recall, fbeta = compute_model_metrics(y_test, inference(model, X_test))
logger.info(
    "Test set - Precision: %s, Recall: %s, F-Beta: %s", precision, recall, fbeta
)


# Calculate metrics in slices

output_list = []
for ed_name in sorted(education_set):
    output_list.append((ed_name, *metrics(data, ed_name, lb)))

output = '\n'.join([' '.join([str(char) for char in line]) for line in output_list])
with open('slice_output.txt', 'w') as f:
    f.write(str(output))

import os

from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

import ml.model
from ml.data import process_data
from ml.const import cat_features

app = FastAPI()


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("Pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


@app.get("/")
async def root():
    return {"message": "Salary predictions API"}


class Inference(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
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
        }


@app.post("/inference/")
async def inference(data: Inference):
    model = load('model/random_forest.joblib')
    encoder = load('model/encoder.joblib')
    df = pd.DataFrame(dict((k, [v]) for k, v in data.__dict__.items()))
    features = [c.replace('-', '_') for c in cat_features]
    X, _, _, _ = process_data(df, categorical_features=features, training=False, encoder=encoder)
    salary_class = ml.model.inference(model, X)[0]
    salary = None
    if salary_class == 0:
        salary = '<=50K'
    else:
        salary = '>50K'
    return {'salary': salary}

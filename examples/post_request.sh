#!/bin/bash

curl -X 'POST' \
  'http://pakallis-udacity.herokuapp.com/inference/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 52,
  "workclass": "Self-emp-inc",
  "fnlgt": 287927,
  "education": "HS-grad",
  "education_num": 9,
  "marital_status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Wife",
  "race": "White",
  "sex": "Female",
  "capital_gain": 15024,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}'
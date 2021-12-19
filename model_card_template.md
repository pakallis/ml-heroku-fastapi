# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

RandomForestClassifier from sklearn with default params.
Created by Pavlos Kallis. 

## Intended Use

Predicts Income based on Demographical characteristics


## Training Data

Obtained from the sample data in https://github.com/udacity/nd0821-c3-starter-code.git
Removed spaces


## Evaluation Data

Obtained from the sample data in https://github.com/udacity/nd0821-c3-starter-code.git
Used 80% of the dataset for training and 20% for testing.
Didn't use stratification

## Metrics

Test set:

Precision: 0.9998404849258254, Recall: 1.0, F-Beta: 0.9999202361011406

Check also `slice_output.txt` for performance in key slices.

## Ethical Considerations

We should be careful about our predictions to not give the wrong impression,
especially for race and sex variables.


## Caveats and Recommendations

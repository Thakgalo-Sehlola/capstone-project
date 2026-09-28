# Model 1 Performance: Logistic Regression

## 1. Overview

This document reports the validation and test-set performance of the Logistic Regression classifier developed for the KKBox churn-prediction experiment.

**Client:** STADIOchoice  
**Dataset used for modelling:** KKBox Churn Prediction Challenge  
**Model:** Logistic Regression (L2 regularisation)  
**Target variable:** `is_churn`

The model is trained and evaluated using KKBox data. The results do not represent STADIOchoice subscriber performance.

## 2. Experimental Configuration

| Parameter | Configuration |
|---|---|
| Algorithm | Logistic Regression |
| Solver | `saga` |
| Penalty | L2 |
| Class weighting | `balanced` |
| Candidate C values | 0.1, 1, 10 |
| Selected C | 1 |
| Maximum iterations | 2,000 |
| Tolerance | 0.001 |
| Random seed | 42 |
| Threshold selection | Validation F1-score maximisation, with precision as tie-break criterion |
| Selected classification threshold | 0.981076 |

The regularisation parameter was selected using validation Average Precision.

The classification threshold was selected using the validation set and was not selected using test-set performance.

## 3. Validation Results

The validation set contains 148,940 customer records, including 9,521 churn records and 139,419 non-churn records.

The following metrics were calculated using the selected model and classification threshold.

| Metric | Validation result |
|---|---:|
| Average Precision (AP) | 0.723098 |
| ROC-AUC | 0.967762 |
| Precision | 0.671042 |
| Recall | 0.719462 |
| F1-score | 0.694409 |
| Balanced Accuracy | 0.847688 |
| Log Loss | 0.323401 |

### Validation Confusion Matrix

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 136,061 | 3,358 |
| Actual churn | 2,671 | 6,850 |

The validation results were used for model selection and threshold determination.

## 4. Test Results

The held-out test set contains 148,940 customer records, including 9,520 churn records and 139,420 non-churn records.

The test set was not used to select the model configuration or classification threshold.

| Metric | Test result |
|---|---:|
| Average Precision (AP) | 0.710419 |
| ROC-AUC | 0.966862 |
| Precision | 0.659078 |
| Recall | 0.711555 |
| F1-score | 0.684312 |
| Balanced Accuracy | 0.843211 |
| Log Loss | 0.329232 |

### Test Confusion Matrix

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 135,916 | 3,504 |
| Actual churn | 2,746 | 6,774 |

### Confusion Matrix Interpretation

- True negatives: 135,916 non-churn customers correctly classified.
- False positives: 3,504 non-churn customers classified as churn.
- False negatives: 2,746 churn customers classified as non-churn.
- True positives: 6,774 churn customers correctly classified.

Precision represents the proportion of customers predicted as churn who were actually labelled as churn.

Recall represents the proportion of actual churn customers correctly identified by the model.

F1-score is the harmonic mean of precision and recall.

Balanced accuracy is the arithmetic mean of the true-positive rate and true-negative rate.

Average Precision summarises the precision-recall curve, while ROC-AUC measures discrimination across classification thresholds.

Log Loss evaluates the quality of the predicted probabilities.

## 5. Validation and Test Comparison

| Metric | Validation | Test |
|---|---:|---:|
| Average Precision | 0.723098 | 0.710419 |
| ROC-AUC | 0.967762 | 0.966862 |
| Precision | 0.671042 | 0.659078 |
| Recall | 0.719462 | 0.711555 |
| F1-score | 0.694409 | 0.684312 |
| Balanced Accuracy | 0.847688 | 0.843211 |
| Log Loss | 0.323401 | 0.329232 |

The validation and test results are relatively close for the reported metrics. This describes performance on the specified split and does not establish generalisation to future customer populations.

## 6. Interpretation and Limitations

The Logistic Regression model provides a baseline for comparison with the XGBoost classifier.

Its test-set Average Precision is 0.710419, and its test-set F1-score at the selected validation threshold is 0.684312.

The model was evaluated on a stratified random split of the KKBox dataset. The test results should not be interpreted as evidence of performance on STADIOchoice subscribers.

The feature window covers December 2016 to February 2017. The exact temporal relationship between the feature window and the competition's churn-label construction has not been independently established. This remains a methodological limitation when interpreting the results.

The model's results also do not establish retention improvement, revenue impact, or operational suitability for STADIOchoice.

## 7. Reproducibility

The model implementation is:

`src/model1_logistic_regression.py`

The preprocessing workflow is:

`src/preprocessing.py`

The fitted model is generated locally at:

`models/logistic_regression.joblib`

The experiment configuration, metrics, predictions, and coefficients are stored locally under:

`experiments/results/`

These generated artifacts are excluded from Git and must be regenerated using the documented pipeline.

## 8. Related Documentation

- [Preprocessing](Preprocessing.md)
- [Feature Engineering](FeatureEngineering.md)
- [Model 2 – XGBoost](Model2.md)
- [Model 2 Performance](Model2Performance.md)
- [Model Comparison](Comparison.md)
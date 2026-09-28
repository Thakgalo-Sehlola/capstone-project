# Model 2 Performance: XGBoost

## 1. Overview

This document reports the validation and test-set performance of the XGBoost classifier developed for the KKBox churn-prediction experiment.

**Client:** STADIOchoice  
**Dataset used for modelling:** KKBox Churn Prediction Challenge  
**Model:** XGBoost binary classifier  
**Target variable:** `is_churn`

The model is trained and evaluated using KKBox data. The results do not represent STADIOchoice subscriber performance.

## 2. Experimental Configuration

| Parameter | Configuration |
|---|---|
| Objective | Binary logistic classification |
| Tree construction | Histogram-based |
| Maximum tree depth candidates | 4, 6 |
| Learning-rate candidates | 0.05, 0.1 |
| Number of estimators | 400 |
| Subsample | 0.8 |
| Column subsample | 0.8 |
| Selected maximum depth | 6 |
| Selected learning rate | 0.05 |
| Scale positive weight | 14.643732 |
| Threshold selection | Validation F1-score maximisation, with precision as tie-break criterion |
| Selected classification threshold | 0.8810855150 |

The selected configuration was determined using validation Average Precision.

The positive-class weight was calculated using the ratio of non-churn to churn observations in the training set.

The classification threshold was selected using validation data and was not selected using test-set performance.

## 3. Validation Results

The validation set contains 148,940 customer records, including 9,521 churn records and 139,419 non-churn records.

| Metric | Validation result |
|---|---:|
| Average Precision (AP) | 0.864812 |
| ROC-AUC | 0.985993 |
| Precision | 0.718636 |
| Recall | 0.850121 |
| F1-score | 0.778868 |
| Balanced Accuracy | 0.913695 |
| Log Loss | 0.142248 |

### Validation Confusion Matrix

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 136,250 | 3,169 |
| Actual churn | 1,427 | 8,094 |

The validation results were used for model configuration selection and threshold determination.

## 4. Test Results

The held-out test set contains 148,940 customer records, including 9,520 churn records and 139,420 non-churn records.

The test set was not used to select the model configuration or classification threshold.

| Metric | Test result |
|---|---:|
| Average Precision (AP) | 0.862505 |
| ROC-AUC | 0.985776 |
| Precision | 0.718033 |
| Recall | 0.845273 |
| F1-score | 0.776475 |
| Balanced Accuracy | 0.911304 |
| Log Loss | 0.143685 |

### Test Confusion Matrix

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 136,260 | 3,160 |
| Actual churn | 1,473 | 8,047 |

### Confusion Matrix Interpretation

- True negatives: 136,260 non-churn customers correctly classified.
- False positives: 3,160 non-churn customers classified as churn.
- False negatives: 1,473 churn customers classified as non-churn.
- True positives: 8,047 churn customers correctly classified.

Precision represents the proportion of customers predicted as churn who were actually labelled as churn.

Recall represents the proportion of actual churn customers correctly identified by the model.

F1-score is the harmonic mean of precision and recall.

Balanced accuracy is the arithmetic mean of the true-positive rate and true-negative rate.

Average Precision summarises the precision-recall curve, while ROC-AUC measures discrimination across classification thresholds.

Log Loss evaluates the quality of the predicted probabilities.

## 5. Validation and Test Comparison

| Metric | Validation | Test |
|---|---:|---:|
| Average Precision | 0.864812 | 0.862505 |
| ROC-AUC | 0.985993 | 0.985776 |
| Precision | 0.718636 | 0.718033 |
| Recall | 0.850121 | 0.845273 |
| F1-score | 0.778868 | 0.776475 |
| Balanced Accuracy | 0.913695 | 0.911304 |
| Log Loss | 0.142248 | 0.143685 |

The validation and test results are close for the reported metrics. This describes performance on the specified split and does not establish generalisation to future customer populations.

## 6. Interpretation and Limitations

The XGBoost model records a test-set Average Precision of 0.862505 and an F1-score of 0.776475 at the selected validation threshold.

Its performance is compared with the Logistic Regression baseline using the same held-out KKBox test set.

The model was evaluated on a stratified random split of the KKBox dataset. The test results should not be interpreted as evidence of performance on STADIOchoice subscribers.

The feature window covers December 2016 to February 2017. The exact temporal relationship between the feature window and the competition's churn-label construction has not been independently established. This remains a methodological limitation when interpreting the results.

The model's results do not establish retention improvement, revenue impact, or operational suitability for STADIOchoice.

## 7. Reproducibility

The model implementation is:

`src/model2_xgboost.py`

The preprocessing workflow is:

`src/preprocessing.py`

The fitted model is generated locally at:

`models/xgboost_pipeline.joblib`

The experiment configuration, metrics, predictions, and feature importance outputs are stored locally under:

`experiments/results/`

These generated artifacts are excluded from Git and must be regenerated using the documented pipeline.

## 8. Related Documentation

- [Preprocessing](Preprocessing.md)
- [Feature Engineering](FeatureEngineering.md)
- [Model 1 – Logistic Regression](Model1.md)
- [Model 1 Performance](Model1Performance.md)
- [Model Comparison](Comparison.md)
# Model Comparison: Logistic Regression vs XGBoost

## 1. Overview

This document compares the Logistic Regression baseline and XGBoost classifier developed for the KKBox churn-prediction experiment.

**Client:** STADIOchoice  
**Dataset used for modelling:** KKBox Churn Prediction Challenge  
**Target variable:** `is_churn`

Both models were trained and evaluated using the KKBox dataset and the same train, validation, and test assignments.

The comparison assesses predictive performance on the specified held-out KKBox test set. It does not establish performance or business impact for STADIOchoice subscribers.

## 2. Experimental Setup

The modelling dataset contains 992,931 labelled customer records.

The target distribution is:

| Target | Number of records | Percentage |
|---|---:|---:|
| Non-churn | 929,460 | 93.61% |
| Churn | 63,471 | 6.39% |
| Total | 992,931 | 100.00% |

The data were divided using a stratified random split with a fixed random seed of 42.

| Split | Total records | Churn records | Non-churn records |
|---|---:|---:|---:|
| Training | 695,051 | 44,430 | 650,621 |
| Validation | 148,940 | 9,521 | 139,419 |
| Test | 148,940 | 9,520 | 139,420 |

Both models use the same split assignments.

Model configuration and classification thresholds were selected using the validation set. The test set was reserved for final evaluation and comparison.

The test-set churn prevalence is approximately 6.39%.

## 3. Models Compared

### Model 1: Logistic Regression

The baseline model uses L2 regularisation, the `saga` solver, and class-balanced weighting.

The selected regularisation parameter is `C = 1`.

The classification threshold was selected using validation F1-score maximisation, with precision as the tie-break criterion.

The selected threshold is 0.981076.

### Model 2: XGBoost

The XGBoost model uses histogram-based tree construction with a maximum depth of 6, a learning rate of 0.05, and 400 estimators.

The model uses a positive-class weight of approximately 14.643732, calculated from the training-set class ratio.

The classification threshold was selected using validation F1-score maximisation, with precision as the tie-break criterion.

The selected threshold is 0.8810855150.

The detailed configurations are documented in:

- [Model 1 – Logistic Regression](Model1.md)
- [Model 2 – XGBoost](Model2.md)

## 4. Test-Set Performance Comparison

The following table reports the observed performance of each fitted model on the same held-out test set.

| Metric | Logistic Regression | XGBoost | Difference (XGBoost minus Logistic Regression) |
|---|---:|---:|---:|
| Average Precision | 0.710419 | 0.862505 | +0.152086 |
| ROC-AUC | 0.966862 | 0.985776 | +0.018914 |
| Precision | 0.659078 | 0.718033 | +0.058956 |
| Recall | 0.711555 | 0.845273 | +0.133718 |
| F1-score | 0.684312 | 0.776475 | +0.092164 |
| Balanced Accuracy | 0.843211 | 0.911304 | +0.068093 |
| Log Loss | 0.329232 | 0.143685 | -0.185547 |

For Average Precision, ROC-AUC, precision, recall, F1-score, and balanced accuracy, a positive difference indicates a higher observed value for XGBoost.

For Log Loss, a negative difference indicates a lower observed loss for XGBoost.

The results describe the two fitted models on the specified KKBox test set. They do not establish the performance of either model on STADIOchoice subscribers.

## 5. Test-Set Confusion Matrices

### Logistic Regression

The classification threshold was selected using validation data.

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 135,916 | 3,504 |
| Actual churn | 2,746 | 6,774 |

### XGBoost

The classification threshold was selected using validation data.

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 136,260 | 3,160 |
| Actual churn | 1,473 | 8,047 |

The confusion matrices show the numbers of correctly and incorrectly classified records at the respective validation-selected thresholds.

The thresholds differ between the models. Therefore, precision, recall, F1-score, and balanced accuracy are comparisons at the selected operating thresholds rather than comparisons at an identical probability cutoff.

## 6. Paired Bootstrap Analysis

### 6.1 Purpose

A paired bootstrap analysis was conducted to estimate uncertainty in selected differences between the two models' test-set performance metrics.

The analysis focuses on:

- Average Precision.
- ROC-AUC.
- F1-score.

The paired approach evaluates both models on the same resampled test observations, preserving the correspondence between their predictions.

### 6.2 Resampling Procedure

The test set contains 9,520 churn records and 139,420 non-churn records.

The bootstrap procedure resamples positive and negative test observations separately, preserving the original class counts within each resample.

The same resampled indices are applied to the predictions from both models.

For each bootstrap resample, the selected metric is calculated for each model, and the difference is calculated as:

`Difference = XGBoost metric - Logistic Regression metric`

The 95% confidence intervals are obtained from the distribution of the paired bootstrap differences.

This procedure estimates uncertainty conditional on the fitted models and the observed test dataset.

It does not account for all sources of uncertainty, including alternative training samples, feature-engineering decisions, model-selection variability, or changes in the future customer population.

### 6.3 Bootstrap Results

| Metric difference | Observed difference | 95% bootstrap confidence interval |
|---|---:|---:|
| Average Precision | +0.152086 | [0.143860, 0.159625] |
| ROC-AUC | +0.018914 | [0.017569, 0.020227] |
| F1-score | +0.092164 | [0.085807, 0.098407] |

The reported intervals are above zero for all three selected metric differences.

This indicates that, under the implemented paired bootstrap procedure and conditional on the specified test set and fitted models, the observed differences remain positive across the reported 95% confidence intervals.

The intervals should not be interpreted as guarantees of future performance or as evidence of performance on STADIOchoice subscribers.

## 7. Interpretation

On the specified KKBox test set, XGBoost records higher Average Precision, ROC-AUC, precision, recall, F1-score, and balanced accuracy than the Logistic Regression baseline.

XGBoost also records lower Log Loss on the same test set.

The paired bootstrap intervals quantify uncertainty in the selected test-set differences, conditional on the implemented methodology.

These findings support a comparison of the two fitted classifiers under the documented KKBox experimental setup.

They do not establish that XGBoost will perform similarly on STADIOchoice subscriber data, nor do they demonstrate that either model will reduce churn, improve retention, or increase revenue.

The current test set is a stratified random holdout rather than a temporal holdout. The feature window covers December 2016 to February 2017, and the exact temporal relationship between the feature window and churn-label construction has not been independently established.

These limitations must be considered when interpreting the results.

## 8. Reproducibility

The model comparison is implemented in:

`src/compare_models.py`

The comparison was run using:

```powershell
python src/compare_models.py --actual-col actual_is_churn --pred-col predicted_is_churn
```

The generated outputs are stored locally under:

`experiments/results/`

The comparison outputs include:

- `model_comparison.json`
- `model_comparison_metrics.csv`
- `model_comparison_bootstrap.csv`

The model-specific test predictions and configurations are also generated locally under `experiments/results/`.

These generated files are excluded from Git and must be regenerated using the documented pipeline.

## 9. Related Documentation

- [Preprocessing](Preprocessing.md)
- [Feature Engineering](FeatureEngineering.md)
- [Model 1 – Logistic Regression](Model1.md)
- [Model 1 Performance](Model1Performance.md)
- [Model 2 – XGBoost](Model2.md)
- [Model 2 Performance](Model2Performance.md)
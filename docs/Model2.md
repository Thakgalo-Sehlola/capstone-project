
# Model 2: XGBoost Classifier

## 1. Model objective

The objective is to develop a supervised binary classification model that estimates the probability of customer churn using engineered KKBox customer features.

XGBoost is implemented as the second model and is evaluated against the Logistic Regression baseline using the same customer cohort, saved data partitions, and evaluation metrics.

The target variable is `is_churn`:

- 0: Non-churn
- 1: Churn

The model produces a churn probability and a threshold-based binary classification.

## 2. Algorithm

XGBoost is a gradient-boosted decision-tree algorithm. It builds an ensemble of decision trees sequentially, with each additional tree contributing to the optimisation of the objective function.

For binary classification, the model estimates the probability of churn using a logistic objective.

The model can capture nonlinear relationships and interactions between predictors through its tree structure.

The implementation uses the XGBoost classifier with the `binary:logistic` objective and histogram-based tree construction.

## 3. Input features

The model uses the same engineered predictors and customer cohort as Model 1.

The features cover:

1. Listening behaviour for December 2016, January 2017 and February 2017.
2. Transaction activity and payment characteristics.
3. Customer demographic and registration characteristics.

The customer identifier `msno` is excluded from the predictors. The target `is_churn` is used as the classification label.

The feature-engineering window is December 2016 to February 2017. The exact temporal relationship between this period and the competition's churn-label construction has not been independently established. Consequently, the model is not claimed to be conclusively free of temporal leakage.

The feature definitions and engineering procedures are documented in the project's data preparation documentation and source code.

## 4. Preprocessing

The model reuses the shared preprocessing implementation in `src/preprocessing.py`.

Numerical predictors:

- Missing values are imputed using training-set medians.
- Numerical variables are not standardised.

Categorical predictors:

- Missing values are represented by `__MISSING__`.
- One-hot encoding is applied, with unknown categories ignored.

All learned preprocessing transformations are fitted on the training partition only.

Class imbalance is addressed using `scale_pos_weight`, calculated as the ratio of non-churn to churn observations in the training partition.

The complete preprocessing and partitioning methodology is documented in [Preprocessing.MD](Preprocessing.MD).

## 5. Experimental setup

### 5.1 Data partitions

The modelling cohort contains 992,931 customers, of whom 63,471 (6.39%) are churners.

| Partition | Customers | Churners | Churn rate |
|---|---:|---:|---:|
| Training | 695,051 | 44,430 | 6.39% |
| Validation | 148,940 | 9,521 | 6.39% |
| Test | 148,940 | 9,520 | 6.39% |

The partitions were created using stratified random sampling with random seed 42.

Both models use the same saved customer-to-partition assignments.

### 5.2 Hyperparameter selection

Three predefined configurations were compared using validation average precision.

| Candidate | Maximum depth | Learning rate | Estimators | Subsample | Column subsampling |
|---|---:|---:|---:|---:|---:|
| 1 | 4 | 0.05 | 400 | 0.8 | 0.8 |
| 2 | 6 | 0.05 | 400 | 0.8 | 0.8 |
| 3 | 4 | 0.10 | 400 | 0.8 | 0.8 |

The candidate with the highest validation average precision was selected.

Average precision was used as the primary selection metric because churn is the minority class and the metric summarises precision across recall increments.

### 5.3 Class imbalance adjustment

The `scale_pos_weight` parameter was calculated using the training partition:

\[
\text{scale\_pos\_weight}=
\frac{\text{Number of non-churn training observations}}
{\text{Number of churn training observations}}
\]

The resulting value was 14.643732.

This increases the relative contribution of positive-class observations to the training objective. It does not change the observed class distribution of the validation or test datasets.

### 5.4 Classification threshold

The classification threshold was selected by maximising F1 on the validation partition.

Where multiple thresholds achieved the same F1, the threshold with the highest precision was selected.

The selected threshold was applied unchanged to the test partition. Test labels were not used for hyperparameter or threshold selection in the final evaluation procedure.

## 6. Evaluation metrics

The model was evaluated using the same metrics as Model 1.

| Metric | Purpose |
|---|---|
| Precision | Proportion of predicted churners who actually churned |
| Recall | Proportion of actual churners detected |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ranking discrimination across classification thresholds |
| Average precision | Summary of precision-recall performance |
| Balanced accuracy | Average of sensitivity and specificity |
| Log loss | Quality of predicted probabilities |
| Confusion matrix | Counts of true negatives, false positives, false negatives and true positives |

Accuracy is not used as the primary selection metric because 93.61% of the cohort belongs to the non-churn class.

The metric definitions and final performance results are presented in [Model2Performance.MD](Model2Performance.MD).

## 7. Model selection results

The selected configuration had maximum depth 6, learning rate 0.05 and 400 estimators.

It achieved validation average precision of 0.864812 and validation ROC-AUC of 0.985993.

The selected classification threshold was 0.881086, determined by the validation F1 procedure described in Section 5.4.

### Selected configuration

| Parameter | Value |
|---|---|
| Objective | `binary:logistic` |
| Tree method | `hist` |
| Maximum depth | 6 |
| Learning rate | 0.05 |
| Number of estimators | 400 |
| Subsample | 0.8 |
| Column subsampling | 0.8 |
| `scale_pos_weight` | 14.643732 |
| Random state | 42 |

## 8. Final model performance

The selected XGBoost model achieved the following results:

| Metric | Validation | Test |
|---|---:|---:|
| Precision | 0.718636 | 0.718033 |
| Recall | 0.850121 | 0.845273 |
| F1-score | 0.778868 | 0.776475 |
| Balanced accuracy | 0.913695 | 0.911304 |
| ROC-AUC | 0.985993 | 0.985776 |
| Average precision | 0.864812 | 0.862505 |
| Log loss | 0.142248 | 0.143685 |

The final test confusion matrix was:

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 136,260 | 3,160 |
| Actual churn | 1,473 | 8,047 |

The final test results are documented and interpreted in [Model2Performance.MD](Model2Performance.MD).

## 9. Implementation and execution

Implementation: `src/model2_xgboost.py`

Run from the repository root:

```powershell
python src/model2_xgboost.py
```

The script loads the prepared feature dataset and shared split assignments, fits the candidate configurations, selects the configuration using validation average precision, selects the classification threshold using validation F1, and evaluates the selected model on the test partition.

The script uses the shared preprocessing functions in `src/preprocessing.py`.

## 10. Saved outputs

| File | Description |
|---|---|
| `models/xgboost_pipeline.joblib` | Fitted preprocessing and XGBoost pipeline |
| `experiments/results/model2_metrics.json` | Validation and test performance metrics |
| `experiments/results/model2_test_predictions.csv` | Test customer IDs, actual labels and predicted probabilities/classifications |
| `experiments/results/model2_feature_importance.csv` | Model-derived feature importance values |
| `experiments/results/model2_configuration.json` | Selected configuration and model parameters |

Feature importance describes model-based predictive contribution and does not establish causal effects.

## 11. Limitations

1. Hyperparameter selection was limited to three predefined configurations and does not establish a globally optimal XGBoost configuration.
2. Classification metrics depend on the selected threshold and the class distribution of the evaluated dataset.
3. Feature importance represents model-based importance, not causal effects.
4. The practical value of churn predictions depends on the costs of false positives and false negatives and the effectiveness of retention interventions, which have not been evaluated.
5. The competition's label-generation timing has not been independently verified against the feature cutoff. Consequently, the model is not claimed to be conclusively free of temporal leakage.

The model comparison, including the paired comparison of test-set metrics, is presented in [Comparison.MD](Comparison.MD).
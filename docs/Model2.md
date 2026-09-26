# Model 2: XGBoost Classifier

## 1. Model objective

XGBoost was implemented as the second churn classification model. Its performance is evaluated against the Logistic Regression baseline using the same customer cohort, saved data partitions, and evaluation metrics.

## 2. Methodology

The model uses the existing 70% training, 15% validation and 15% test partitions. Numerical predictors were median-imputed, while categorical predictors were one-hot encoded. Preprocessing was fitted on the training data only.

Class imbalance was addressed using the `scale_pos_weight` parameter, calculated as the ratio of non-churn to churn cases in the training partition.

Three configurations were compared using validation average precision:

| Candidate | Max depth | Learning rate | Estimators |
| --------- | --------: | ------------: | ---------: |
| 1         |         4 |          0.05 |        400 |
| 2         |         6 |          0.05 |        400 |
| 3         |         4 |          0.10 |        400 |

The selected configuration was the candidate with the highest validation average precision. The classification threshold was selected on the validation set by maximising F1-score, using precision as a tie-breaker. The selected threshold was then applied unchanged to the test set.

## 3. Selected configuration

| Parameter                | Value                          |
| ------------------------ | ------------------------------ |
| Objective                | Binary logistic classification |
| Tree method              | Histogram                      |
| Maximum depth            | 6                              |
| Learning rate            | 0.05                           |
| Number of estimators     | 400                            |
| Subsample                | 0.8                            |
| Column subsampling       | 0.8                            |
| Scale positive weight    | 14.643732                      |
| Random state             | 42                             |
| Classification threshold | 0.881086                       |

## 4. Results

### Validation performance

| Metric            |   Result |
| ----------------- | -------: |
| Average precision | 0.864812 |
| ROC-AUC           | 0.985993 |
| Precision         | 0.718636 |
| Recall            | 0.850121 |
| F1-score          | 0.778868 |
| Balanced accuracy | 0.913695 |
| Log loss          | 0.142248 |

### Test performance

| Metric            |   Result |
| ----------------- | -------: |
| Average precision | 0.862505 |
| ROC-AUC           | 0.985776 |
| Precision         | 0.718033 |
| Recall            | 0.845273 |
| F1-score          | 0.776475 |
| Balanced accuracy | 0.911304 |
| Log loss          | 0.143685 |

The test confusion matrix contained 136,260 true negatives, 3,160 false positives, 1,473 false negatives and 8,047 true positives.

The test average precision was 0.862505, while test ROC-AUC was 0.985776. At the selected threshold, the model correctly classified 8,047 of the 9,520 churners in the test partition.

## 5. Limitations

The hyperparameter search was limited to three configurations and does not establish a globally optimal XGBoost configuration.

The reported classification metrics depend on the selected threshold. Business usefulness would also depend on the cost of contacting non-churners, the cost of missing churners, and the effectiveness of retention interventions.

Feature importance represents model-based importance, not causal effects. It should not be interpreted as evidence that changing a feature will prevent churn.

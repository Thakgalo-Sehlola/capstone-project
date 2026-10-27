# STADIOchoice Customer Churn Prediction

## STADIO CAP182 Data Science Capstone

**Client:** STADIOchoice  
**Project:** Customer Churn Prediction  
**Programme:** Postgraduate Diploma in Data Science
**Dataset used for modelling:** KKBox Churn Prediction Challenge (public dataset)

---

# 1. Project Overview

## 1.1 Business Context: STADIOchoice

STADIOchoice is the client for this capstone project. The business problem concerns customer churn, specifically the identification of subscribers who may discontinue their subscriptions.

Customer churn can affect subscription revenue, customer retention, and the sustainability of subscription-based services. Identifying subscribers who are at risk of churning may support more informed retention planning and customer engagement.

The proposed analytical solution for STADIOchoice is a supervised machine-learning classification model that estimates the probability of subscriber churn using historical subscriber, subscription, transaction, and behavioural information.

The intended business application is to support the identification of subscribers who may require further retention attention. Any actual retention intervention, deployment, or assessment of business impact would require STADIOchoice-specific data, validation, and operational evaluation.

**Important distinction:** STADIOchoice is the client and the subject of the business problem. The models developed and evaluated in this repository use the public KKBox dataset, not STADIOchoice subscriber records.

## 1.2 Analytical Implementation: KKBox

The KKBox Churn Prediction Challenge dataset is used as a publicly available dataset to implement and evaluate a customer churn classification proof of concept.

KKBox provides labelled subscription-related customer data and associated transaction, membership, and listening-behaviour information. These data allow the project to demonstrate data preparation, feature engineering, classification modelling, model evaluation, and statistical comparison.

The project uses two supervised binary classification models:

1. Logistic Regression: the baseline classification model
2. XGBoost: the non-linear tree-based classification model

Both models are evaluated using a common training, validation, and test split. Model configuration and classification thresholds are selected using validation data, while the held-out test set is used for final performance evaluation and model comparison.

The analytical deliverable is a reproducible KKBox-based churn classification experiment, supported by data-engineering scripts, trained models, evaluation results, and statistical comparison outputs.

The KKBox results demonstrate performance on the specified public dataset and experimental setup. They do not establish how either model would perform on STADIOchoice subscribers.

---

# 2. Business Problem and Analytical Objectives

## 2.1 STADIOchoice Business Problem

The business problem is to identify subscribers who may discontinue their subscriptions so that STADIOchoice can investigate and potentially prioritise appropriate retention activities.

The proposed business question is:

> Can historical subscriber, subscription, transaction, and behavioural data be used to develop a supervised classification model that identifies subscribers at risk of churn and supports customer-retention decision-making at STADIOchoice?

The question is framed around STADIOchoice's business context. The current modelling experiment uses KKBox as a public proxy dataset because STADIOchoice subscriber data have not been used to train or evaluate the models in this repository.

## 2.2 Analytical Objectives

The project has the following objectives:

1. Establish a reproducible data-processing workflow for a large customer-behaviour dataset.
2. Engineer customer-level features from membership, transaction, and listening-behaviour data.
3. Develop a Logistic Regression baseline and an XGBoost binary classification model using the KKBox dataset.
4. Compare the models using classification and probability-based evaluation metrics on a held-out test set.
5. Quantify uncertainty in selected test-set performance differences using paired bootstrap confidence intervals.
6. Document the data requirements and methodological considerations for future STADIOchoice-specific model development.

## 2.3 Scope and Limitations

The current implementation is a proof of concept using KKBox data.

It does not include:

- Training or testing on actual STADIOchoice subscriber data.
- Verification of churn predictions against STADIOchoice's historical subscriber outcomes.
- Deployment into STADIOchoice's operational systems.
- A controlled evaluation of retention campaigns or interventions.
- Measurement of actual improvements in subscriber retention, revenue, or customer lifetime value.

These activities would require additional client data, business definitions, and validation.

---

# 3. Data Sources and Data Acquisition

## 3.1 Public Dataset Used for the Experiment

The modelling experiment uses the following public dataset:

**KKBox Churn Prediction Challenge**

Dataset source: [Kaggle – KKBox Churn Prediction Challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data)

The dataset is used for the technical implementation and evaluation of the churn-prediction proof of concept.

The relevant input files are:

| File | Purpose |
|---|---|
| `train.csv` | Provides labelled customer records and the churn target used to construct the modelling dataset. |
| `members_v3.csv` | Provides customer membership and demographic attributes. |
| `transactions.csv` | Provides subscription transaction and payment-related information. |
| `user_logs.csv.7z` | Compressed listening-behaviour records used to engineer behavioural features. |

The listening-log archive is processed using the project's streaming and chunked-processing workflow.

The exact source files and their required formats should be obtained from the official Kaggle competition data page. Users must comply with the applicable competition rules and data-access conditions.

## 3.2 Obtaining the Data

The raw KKBox data are not committed to this Git repository.

To reproduce the experiment:

1. Create or sign in to a Kaggle account.
2. Open the [KKBox Churn Prediction Challenge data page](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data).
3. Review and accept the applicable competition rules and data-access conditions.
4. Download the required files.
5. Place the source files in the following local directory structure.

```text
data/
└── kkbox/
    └── raw/
        ├── train.csv
        ├── members_v3.csv
        ├── transactions.csv
        └── user_logs.csv.7z
```

The filename of the downloaded listening-log archive should be checked against the actual Kaggle download. If the archive has a different name or format, the local file and processing configuration must be adjusted accordingly.

The raw data are excluded from Git because of their size and are managed separately from the source code. A fresh repository clone will not contain the raw KKBox data.

The processed feature datasets, split assignments, fitted models, and experiment results are also generated locally and are excluded from version control.

## 3.3 STADIOchoice Client Data

The KKBox dataset is not a substitute for STADIOchoice subscriber data.

For the intended client-specific application, STADIOchoice would need to provide data corresponding to its own subscribers, subscription products, transaction history, customer activity, and churn outcomes.

The proposed client data requirements are documented separately:

- [Part C – STADIOchoice Data Request (PDF)](part_c_data_request.pdf)
- [Part C – STADIOchoice Data Request (Word)](part_c_data_request.docx)

The data request defines the information required to investigate the client problem and support future development of a STADIOchoice-specific churn model.

Until suitable STADIOchoice data are obtained and evaluated, the results in this repository must be interpreted exclusively as KKBox experimental results.

---

# 4. Data Engineering and Processing

## 4.1 Processing Challenge

The KKBox listening-behaviour data presented a substantial computational challenge in the local development environment.

The compressed listening-log archive was approximately 6.7 GB, while the extracted CSV was approximately 30.5 GB. The development laptop had approximately 30 GB of free storage at the time.

The initial approach of processing the listening-log CSV directly with Pandas ran for several hours without completing the required processing. The operation was stopped, and the workflow was redesigned to reduce the need to load the entire dataset into memory.

The revised workflow uses:

- Streaming and chunked processing.
- Parquet files for intermediate and processed datasets.
- DuckDB for analytical queries and aggregations.
- Pandas for operations on appropriately sized data and modelling inputs.

These changes were made to accommodate the available storage, memory, and processing resources.

## 4.2 Why Parquet and DuckDB Were Used

Parquet and DuckDB serve different but complementary purposes.

### Parquet

Parquet is a columnar data-storage format used to persist the processed data and intermediate outputs.

Its use in this project supports:

- Column-oriented storage and selective column reading.
- Storage of structured data with data types.
- Compressed storage of intermediate and engineered datasets.
- Reuse of processed data without repeatedly extracting and parsing the complete raw CSV.
- Chunked conversion of the large listening-log data into manageable files.

The project stores generated behaviour, membership, transaction, and modelling features in Parquet format.

Parquet is a storage format, not a query engine or a guarantee that every operation will fit into memory.

### DuckDB

DuckDB is used as a local analytical query engine for processing and querying the large datasets.

It supports SQL-based aggregations and analytical operations over CSV and Parquet data without requiring the entire raw dataset to be loaded into a Pandas DataFrame.

In this project, DuckDB supports the processing workflow by allowing analytical operations to be performed over the stored data while managing resource usage within the constraints of the development environment.

DuckDB does not eliminate the computational cost of processing the data. Execution time and memory usage remain dependent on the query, data volume, storage performance, and available system resources.

## 4.3 Memory and Resource Constraints

Even after adopting DuckDB and Parquet, some processing operations remained slow and competed with other applications and processes running on the operating system.

The processing configuration was adjusted to limit memory usage to approximately 3 GB for the relevant workload, allowing resources to remain available for other system processes.

This was a local development decision based on the available laptop resources.

The approximately 3 GB limit is not a universal requirement for the project. Other machines may require different resource settings depending on their available RAM, CPU, storage, and concurrent workloads.

The resource constraint can also increase execution time by restricting the memory available to individual operations.

## 4.4 Feature Engineering

The modelling dataset contains one row per labelled customer, identified by `msno`.

The engineered predictors are derived from three broad information sources:

1. Customer membership attributes.
2. Subscription transaction information.
3. Listening-behaviour information.

The feature-engineering implementation is documented in:

- [Feature Engineering](docs/FeatureEngineering.md)
- [Preprocessing](docs/Preprocessing.md)

### Listening-behaviour features

The listening logs are processed using customer-level and monthly aggregations over the following observation windows:

| Window | Period |
|---|---|
| M3 | 1-31 December 2016 |
| M2 | 1-31 January 2017 |
| M1 | 1-28 February 2017 |

The engineered behavioural information includes active listening days, listening duration, recorded unique-song counts, and aggregated listening-event counts.

The total `num_unq` feature represents the sum of daily unique-song counts, not a globally deduplicated count of songs listened to by a customer.

Similarly, the aggregated song-event feature represents the sum of the available listening-threshold counts, not a separately verified count of distinct songs or listening sessions.

### Transaction features

Transaction features are derived from the available subscription transaction records within the defined observation period.

These include transaction counts, payment-related aggregates, subscription-plan information, auto-renewal and cancellation indicators, and transaction recency.

Rates for auto-renewal and cancellation are calculated as the number of transaction records with the corresponding indicator equal to 1 divided by the number of transaction records in the relevant aggregation.

The currency and units of the source transaction amounts have not been independently validated against STADIOchoice business definitions. The engineered transaction values therefore retain the units represented by the KKBox source data.

### Membership features

Membership attributes include available categorical membership fields and derived registration information.

Age values outside the specified valid range of 5-99 are treated as missing. Missing values and categorical encoding are handled in the preprocessing pipeline.

### Temporal limitation

The feature window covers December 2016 to February 2017. March 2017 is excluded from the engineered feature window.

However, the exact temporal relationship between the competition's churn-label construction and the end of the feature window has not been independently established.

The exclusion of March does not, by itself, establish that every predictor was observed before the target outcome became known.

The results must therefore be interpreted with this temporal limitation in mind.

---

# 5. Repository Structure

The repository separates source code, documentation, experimental outputs, and locally generated data.

The following structure describes the tracked project components and the locations used for generated outputs.

```text
capstone-project
├── data
│   ├── kkbox
│   │   ├── processed
│   │   │   ├── log_chunks
│   │   │   └── preprocessing_summary.json
│   │   └── raw
│   │       ├── members_v3.csv
│   │       ├── train.csv
│   │       └── transactions.csv
│   └── README.md
├── docs
│   ├── Comparison.md
│   ├── FeatureEngineering.md
│   ├── Model1.md
│   ├── Model1Performance.md
│   ├── Model2.md
│   ├── Model2Performance.md
│   └── Preprocessing.md
├── experiments
│   ├── results
│   │   ├── model1_coefficients.csv
│   │   ├── model1_configuration.json
│   │   ├── model1_metrics.json
│   │   ├── model1_test_predictions.csv
│   │   ├── model2_configuration.json
│   │   ├── model2_feature_importance.csv
│   │   ├── model2_metrics.json
│   │   ├── model2_test_predictions.csv
│   │   ├── model_comparison.json
│   │   ├── model_comparison_bootstrap.csv
│   │   └── model_comparison_metrics.csv
│   └── setup
│       └── README.md
├── models
│   ├── logistic_regression.joblib
│   ├── README.md
│   └── xgboost_pipeline.joblib
├── notebooks
│   └── README.md
├── reports
│   ├── README.md
│   └── SS2_Part_A_Literature_Review.pdf
├── src
│   ├── comparison
│   │   └── README.md
│   ├── statistical
│   │   └── README.md
│   ├── visualisation
│   │   └── README.md
│   ├── build_model_dataset.py
│   ├── compare_models.py
│   ├── feature_engineering.py
│   ├── member_features.py
│   ├── model1_logistic_regression.py
│   ├── model2_xgboost.py
│   ├── prepare_user_logs.py
│   ├── preprocessing.py
│   ├── profile_churn_timeline.py
│   ├── profile_labelled_expiry.py
│   ├── profile_training_data.py
│   ├── profile_transaction_anomalies.py
│   ├── profile_user_logs.py
│   └── transaction_features.py
├── LICENSE
├── part_c_data_request.docx
├── part_c_data_request.pdf
└── README.md
```

The directory structure above distinguishes source files and documentation from generated artifacts. The locally generated data and experiment outputs may exist on the development machine but are not necessarily present in a fresh Git clone.

The local data structure includes the following generated outputs:

```text
data/
└── kkbox/
    ├── raw/
    │   # Downloaded competition data
    │
    └── processed/
        ├── log_chunks/
        ├── behaviour_features.parquet
        ├── member_features.parquet
        ├── transaction_features.parquet
        ├── kkbox_model_features.parquet
        ├── split_assignments.parquet
        └── preprocessing_summary.json
```

The model and experiment directories contain the locally generated fitted models, model configurations, predictions, metrics, and comparison results.

These generated files are excluded from Git and must be reproduced locally.

---

# 6. Experimental Design

## 6.1 Modelling Dataset

The final engineered modelling dataset is stored locally as:

`data/kkbox/processed/kkbox_model_features.parquet`

The dataset contains:

- 992,931 labelled customer records.
- 32 columns: the customer identifier, churn target, and 30 predictors.

The target variable is `is_churn`, representing the binary churn label supplied by the KKBox competition dataset.

The observed target distribution is:

| Target | Number of customers | Percentage |
|---|---:|---:|
| Non-churn | 929,460 | 93.61% |
| Churn | 63,471 | 6.39% |
| Total | 992,931 | 100.00% |

The target distribution demonstrates class imbalance, with churn representing approximately 6.39% of the labelled records.

This imbalance is relevant to model configuration, threshold selection, and performance evaluation.

These counts describe the KKBox modelling dataset and must not be interpreted as the churn prevalence of STADIOchoice subscribers.

## 6.2 Train, Validation, and Test Split

The dataset is divided using a stratified random split with a fixed random seed of 42.

The split proportions are:

- 70% training.
- 15% validation.
- 15% test.

The split assignments are stored locally in:

`data/kkbox/processed/split_assignments.parquet`

The observed split sizes are:

| Split | Total records | Churn records | Non-churn records |
|---|---:|---:|---:|
| Training | 695,051 | 44,430 | 650,621 |
| Validation | 148,940 | 9,521 | 139,419 |
| Test | 148,940 | 9,520 | 139,420 |

The training set is used to fit the preprocessing transformations and model parameters.

The validation set is used for model configuration selection and classification-threshold selection.

The test set is reserved for final performance evaluation and comparison.

Both models use the same split assignments to support a consistent comparison.

The split is stratified by the churn target. It is not a temporal holdout split, and its results should not be interpreted as evidence of performance under a future-time deployment scenario.

## 6.3 Preprocessing

The preprocessing workflow is implemented in `src/preprocessing.py`.

The 30 raw predictors consist of 26 numerical and 4 categorical predictors.

The categorical predictors are:

- `city`
- `gender`
- `registered_via`
- `registration_month`

The preprocessing pipeline includes:

1. Handling missing and infinite numerical values.
2. Median imputation of numerical predictors, with imputation values fitted on training data.
3. Conversion and imputation of categorical predictors.
4. One-hot encoding of categorical variables, with unseen categories handled through `handle_unknown="ignore"`.
5. Feature scaling for Logistic Regression.

Logistic Regression uses a sparse-compatible StandardScaler configuration with `with_mean=False`.

XGBoost uses the encoded features without the Logistic Regression scaling step.

Preprocessing transformations are fitted on the training data only. Validation and test data are transformed using the fitted training transformations.

The preprocessing and modelling documentation provides the detailed implementation and configuration.

---

# 7. Models and Experimental Results

Two supervised binary classifiers were developed using the KKBox modelling dataset.

The models share the same target variable, preprocessing framework, split assignments, and held-out test set.

The following sections report the actual KKBox experimental results.

## 7.1 Model 1: Logistic Regression

Logistic Regression is used as the baseline binary classification model.

The implementation is contained in:

`src/model1_logistic_regression.py`

The model uses an L2-regularised Logistic Regression classifier with the `saga` solver and class-balanced weighting.

The regularisation parameter was selected from the candidate values 0.1, 1, and 10 using validation average precision.

The selected value was:

`C = 1`

The classification threshold was selected using the validation set by maximising the F1-score, with precision used as a tie-break criterion.

The selected threshold was:

`0.981076`

### Logistic Regression test-set results

| Metric | Test result |
|---|---:|
| Average Precision (AP) | 0.710419 |
| ROC-AUC | 0.966862 |
| Precision | 0.659078 |
| Recall | 0.711555 |
| F1-score | 0.684312 |
| Balanced Accuracy | 0.843211 |
| Log Loss | 0.329232 |

The test-set confusion matrix is:

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 135,916 | 3,504 |
| Actual churn | 2,746 | 6,774 |

The model's detailed configuration and results are documented in:

- [Model 1 – Logistic Regression](docs/Model1.md)
- [Model 1 – Performance](docs/Model1Performance.md)

The fitted model is generated locally and stored at:

`models/logistic_regression.joblib`

## 7.2 Model 2: XGBoost

XGBoost is used as the non-linear tree-based classification model.

The implementation is contained in:

`src/model2_xgboost.py`

The experiment evaluates candidate configurations varying tree depth and learning rate.

The selected configuration uses:

- Maximum tree depth: 6.
- Learning rate: 0.05.
- Number of boosting estimators: 400.
- Subsample: 0.8.
- Column subsample: 0.8.
- Histogram-based tree construction.

The positive-class weighting is configured using the ratio of non-churn to churn observations in the training data.

The resulting `scale_pos_weight` is approximately 14.643732.

The selected configuration is based on validation average precision.

The classification threshold was selected using the validation set by maximising F1-score, with precision used as a tie-break criterion.

The selected threshold was:

`0.8810855150`

### XGBoost test-set results

| Metric | Test result |
|---|---:|
| Average Precision (AP) | 0.862505 |
| ROC-AUC | 0.985776 |
| Precision | 0.718033 |
| Recall | 0.845273 |
| F1-score | 0.776475 |
| Balanced Accuracy | 0.911304 |
| Log Loss | 0.143685 |

The test-set confusion matrix is:

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 136,260 | 3,160 |
| Actual churn | 1,473 | 8,047 |

The model's detailed configuration and results are documented in:

- [Model 2 – XGBoost](docs/Model2.md)
- [Model 2 – Performance](docs/Model2Performance.md)

The fitted model is generated locally and stored at:

`models/xgboost_pipeline.joblib`

## 7.3 Test-Set Model Comparison

The models are compared using the same 148,940 test records, including 9,520 churn records and 139,420 non-churn records.

The following table reports the observed differences between the XGBoost and Logistic Regression test-set metrics.

| Metric | Logistic Regression | XGBoost | XGBoost minus Logistic Regression |
|---|---:|---:|---:|
| Average Precision | 0.710419 | 0.862505 | +0.152086 |
| ROC-AUC | 0.966862 | 0.985776 | +0.018914 |
| Precision | 0.659078 | 0.718033 | +0.058956 |
| Recall | 0.711555 | 0.845273 | +0.133718 |
| F1-score | 0.684312 | 0.776475 | +0.092164 |
| Balanced Accuracy | 0.843211 | 0.911304 | +0.068093 |
| Log Loss | 0.329232 | 0.143685 | -0.185547 |

The reported values show the observed difference between the two fitted classifiers on the specified KKBox test set.

Average Precision is particularly relevant to this experiment because the churn target is imbalanced. Precision, recall, and F1-score are reported at the classification thresholds selected on the validation set.

Log Loss evaluates the predicted probabilities and is reported alongside the threshold-dependent classification metrics.

These results describe the performance of the fitted models under the documented KKBox experimental setup. They do not establish operational performance or business impact for STADIOchoice.

## 7.4 Paired Bootstrap Comparison

A paired bootstrap analysis was conducted to estimate uncertainty in selected test-set performance differences.

The comparison uses the same resampled test observations for both fitted models.

The bootstrap procedure resamples the positive and negative test observations separately, preserving the original class counts within each resample.

The same resampled indices are applied to the predictions from both models, allowing the performance differences to be calculated on matched resamples.

The reported 95% bootstrap confidence intervals for the difference in test-set metrics are:

| Metric difference (XGBoost minus Logistic Regression) | Observed difference | 95% bootstrap confidence interval |
|---|---:|---:|
| Average Precision | +0.152086 | [0.143860, 0.159625] |
| ROC-AUC | +0.018914 | [0.017569, 0.020227] |
| F1-score | +0.092164 | [0.085807, 0.098407] |

The intervals describe uncertainty conditional on the held-out test dataset, fitted models, and implemented bootstrap procedure.

They do not account for all sources of uncertainty, such as variation arising from different training samples, alternative feature windows, or future customer populations.

The detailed methodology and results are documented in:

[Model Comparison](docs/Comparison.md)

The comparison script is:

`src/compare_models.py`

The generated comparison metrics, bootstrap results, and model predictions are stored locally under:

`experiments/results/`

---

# 8. Results Interpretation and STADIOchoice Implications

The experimental results demonstrate that both Logistic Regression and XGBoost can be trained and evaluated as binary churn classifiers on the KKBox dataset using the implemented processing and modelling pipeline.

On the specified KKBox test set, XGBoost records higher Average Precision, ROC-AUC, recall, F1-score, and balanced accuracy than the Logistic Regression baseline, while also recording lower Log Loss.

The paired bootstrap analysis provides intervals for selected differences in test-set performance.

These are experimental findings for KKBox. They are not evidence that XGBoost, or either model, will achieve the same performance on STADIOchoice subscribers.

The results also do not establish that deploying either classifier would reduce actual churn or increase retention. Those outcomes would require client-specific validation and, for intervention impact, an appropriate operational or experimental evaluation.

For STADIOchoice, the next analytical stage would require:

1. Obtaining the relevant STADIOchoice subscriber, subscription, transaction, and behavioural data.
2. Confirming the client's operational definition of churn and the time at which the prediction must be made.
3. Establishing appropriate observation and prediction windows that prevent the use of information unavailable at prediction time.
4. Engineering and validating client-specific features.
5. Developing and evaluating models using suitable STADIOchoice training, validation, and test data.
6. Establishing operational evaluation criteria and business measures before considering deployment.

The KKBox experiment provides the technical foundation and documented methodology for investigating the client problem. It does not replace the need for STADIOchoice-specific data and evaluation.

---

# 9. Reproducing the Project

## 9.1 Environment Setup

The project uses Python 3.13.7 and the dependencies pinned in `requirements.txt`.

From the project root, create and activate a virtual environment, then install the dependencies:

```powershell
py -3.13 -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt.
```

Verify the Python version and dependency compatibility:
```powershell
python --version

python -m pip check
```

The expected Python version is 3.13.7, and `pip check` should report no broken requirements.

## Data Requirements

Before running the pipeline, obtain the required KKBox source files from the KKBox Churn Prediction Challenge and place them in the expected local data directory, following the structure specified in `data/README.md`.

The raw dataset is not included in the repository. Users must obtain it separately, subject to Kaggle's dataset access and usage terms.

## Resource Requirements

The processing scripts use streaming, chunked processing, Parquet files and DuckDB to manage the large dataset. However, processing remains resource-intensive.

Ensure sufficient disk space and system memory for the source data, intermediate files, generated Parquet files, fitted models and experiment outputs. Actual requirements depend on the available hardware and processing configuration.

## 9.2 Processing and Modelling Pipeline

Run the scripts in the order required by their input dependencies.

### Step 1: Prepare the listening logs

```powershell
python src/prepare_user_logs.py
```

This stage processes the compressed listening-log data using the project's streaming and chunked workflow.

The generated chunks are stored under:

`data/kkbox/processed/log_chunks/`

### Step 2: Generate customer features

Run the feature-engineering scripts:

```powershell
python src/member_features.py
python src/transaction_features.py
python src/feature_engineering.py
```

These scripts generate membership, transaction, and behavioural feature datasets.

The outputs include:

- `member_features.parquet`
- `transaction_features.parquet`
- `behaviour_features.parquet`

All generated files are stored locally under `data/kkbox/processed/`.

### Step 3: Build the modelling dataset

```powershell
python src/build_model_dataset.py
```

This stage combines the labelled customer records with the engineered predictors to create:

`data/kkbox/processed/kkbox_model_features.parquet`

### Step 4: Preprocess the modelling dataset

```powershell
python src/preprocessing.py
```

This stage creates the preprocessing summary and train, validation, and test split assignments.

The split assignments are stored at:

`data/kkbox/processed/split_assignments.parquet`

The preprocessing summary is stored at:

`data/kkbox/processed/preprocessing_summary.json`

### Step 5: Train the models

Train the Logistic Regression baseline:

```powershell
python src/model1_logistic_regression.py
```

Train the XGBoost classifier:

```powershell
python src/model2_xgboost.py
```

The fitted models and their associated experiment outputs are generated locally.

### Step 6: Compare the models

```powershell
python src/compare_models.py --actual-col actual_is_churn --pred-col predicted_is_churn
```

This stage performs the documented model comparison and paired bootstrap analysis.

The resulting metrics and comparison outputs are written to the local experiment results directory.

The exact configuration, inputs, outputs, and assumptions for the individual stages are documented in the relevant files under `docs/`, `src/`, `models/`, and `experiments/`.

The commands above describe the intended processing order. They assume the required source files, Python dependencies, and expected script configurations are available. A successful fresh-clone reproduction should be verified against the actual script outputs and the installed dependency versions.

---

# 10. RAAIDD

This section records the actual implementation issues encountered, the responses taken, and the remaining project risks.

Issues describe problems that occurred during the project. Risks describe potential future events or uncertainties that may affect the project or its outcomes.

## 10.1 Risks

| ID | Risk | Mitigation |
|---|---|---|
| R1 | The temporal relationship between the KKBox feature window and churn-label construction has not been independently established. Some transaction features include February 2017 activity. | Verify the competition's label-generation rules and document the prediction-time assumptions. Interpret the current metrics with this limitation in mind. |
| R2 | The KKBox customer population and subscription behaviour may differ from those of STADIOchoice. Model performance may not transfer to the client population. | Treat the current work as a public-dataset proof of concept. Obtain suitable STADIOchoice data and perform client-specific feature engineering, validation, and testing before drawing conclusions about client performance. |
| R3 | The raw data, processed datasets, fitted models, and experiment outputs are excluded from Git. A fresh clone does not contain all the files required to reproduce the experiment immediately. | Document the official data source, required files, setup procedure, and processing pipeline. Regenerate the local artifacts by following the documented workflow. |
| R4 | Limited local storage, memory, CPU availability, and concurrent system workloads may affect processing time and reproducibility across machines. | Use chunked processing, Parquet intermediates, DuckDB resource configuration, and adequate storage. Document machine-specific configuration and avoid assuming identical execution times across environments. |

## 10.2 Actual Issues Encountered

| ID | Actual issue | Response / mitigation | Residual impact |
|---|---|---|---|
| I1 | The initial direct Pandas approach to processing the large listening-log CSV ran for several hours without completing the required operation. | Stopped the run and redesigned the workflow around streaming, chunked processing, Parquet intermediates, and DuckDB-based analytical queries. | Processing remains computationally intensive, but the workflow avoids requiring the entire raw listening-log dataset to be loaded into a single Pandas DataFrame. |
| I2 | The extracted listening-log CSV was approximately 30.5 GB, while the laptop had approximately 30 GB of free storage. This created a practical storage constraint, particularly when accounting for intermediate files and other project data. | Used streaming extraction and chunked conversion to Parquet, while excluding large raw and generated data files from Git. | Adequate local disk space is still required for source archives, processed chunks, engineered features, and generated outputs. |
| I3 | Processing remained slow and resource-intensive even after adopting DuckDB and Parquet. Other operating-system processes and applications also required memory and CPU resources. | Reallocated processing resources and configured a RAM limit of approximately 3 GB for the relevant workload to allow other processes to continue operating. | The memory constraint can increase execution time. Processing remains dependent on available hardware and concurrent workloads. The 3 GB setting is a local development configuration rather than a universal project requirement. |

## 10.3 Dependencies and Assumptions

The project depends on:

- Access to the required public KKBox competition data.
- A Python environment with the dependencies listed in `requirements.txt`.
- Sufficient local storage and computational resources to process the data.
- The availability and compatibility of the source files and script configurations.
- The documented KKBox target and feature definitions for interpretation of the experimental results.

For the intended STADIOchoice application, additional dependencies include access to suitable client data, agreement on the operational definition of churn, and confirmation of the relevant prediction and observation periods.

The absence of STADIOchoice-specific training data means that client-specific model performance and business impact remain unverified.

---

# 11. Project Documentation

The following documentation provides further detail on the project's methodology, implementation, and results.

| Document | Description |
|---|---|
| [Part A - Literature Review](reports/SS2_Part_A_Literature_Review.pdf) | Literature review of relevant churn-prediction research and datasets. |
| [Part C - STADIOchoice Data Request (PDF)](part_c_data_request.pdf) | Proposed client data requirements for the STADIOchoice problem. |
| [Part C - STADIOchoice Data Request (Word)](part_c_data_request.docx) | Editable version of the client data request. |
| [Feature Engineering](docs/FeatureEngineering.md) | Feature definitions, observation windows, aggregation logic, and engineered predictors. |
| [Preprocessing](docs/Preprocessing.md) | Preprocessing transformations, data splits, and missing-value handling. |
| [Model 1 - Logistic Regression](docs/Model1.md) | Baseline model configuration and methodology. |
| [Model 1 - Performance](docs/Model1Performance.md) | Logistic Regression validation and test-set results. |
| [Model 2 - XGBoost](docs/Model2.md) | XGBoost configuration and modelling methodology. |
| [Model 2 - Performance](docs/Model2Performance.md) | XGBoost validation and test-set results. |
| [Model Comparison](docs/Comparison.md) | Comparative metrics and paired bootstrap analysis. |
| [Data Directory](data/README.md) | Data acquisition and local data documentation. |
| [Experiment Setup](experiments/setup/README.md) | Experiment setup documentation. |
| [Model Artifacts](models/README.md) | Model artifact documentation. |

---

# 12. Conclusion

This capstone implements a supervised customer churn classification proof of concept using the publicly available KKBox Churn Prediction Challenge dataset.

The project addresses the practical challenges of processing a large listening-behaviour dataset under limited local computing resources and establishes a reproducible workflow for feature engineering, preprocessing, model training, evaluation, and statistical comparison.

Logistic Regression provides the baseline classifier, while XGBoost provides a non-linear classification approach. Both are evaluated on a common held-out test set, with validation-based model and threshold selection and paired bootstrap analysis of selected performance differences.

The experimental results apply to the KKBox dataset and the documented experimental setup.

STADIOchoice remains the intended client and business context. The KKBox models have not been trained or evaluated on STADIOchoice subscriber records, and their results do not establish client-specific predictive performance or business impact.

The next stage for a client-specific solution is to obtain and validate suitable STADIOchoice data, confirm the operational definition and timing of churn, and conduct model development and evaluation using the client's own subscriber population.
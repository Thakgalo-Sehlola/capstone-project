# STADIOchoice Churn Prediction

## 1. Project overview

This notebook investigates whether subscriber/account information and
behaviour available at a monthly index date can rank eligible
STADIOchoice subscribers by their probability of a qualifying churn
event in the **following 30 days**. The primary model is **XGBoost**,
selected as the model carried forward from the earlier CAP182 work.

The notebook is an end-to-end experiment: it loads and checks the
supplied operational tables, defines a forward-looking target, builds a
subscriber-month panel, engineers predictors from information available
at the index date, applies preprocessing, compares XGBoost
configurations against a simple reference benchmark, selects a
classification threshold using validation data, evaluates the frozen
model on a later test period, and examines feature contributions with
SHAP.

**Overall conclusion:** the notebook completes the modelling and
evaluation workflow, but the final model does **not** demonstrate
dependable churn discrimination on the held-out test period. Test
average precision is only modestly above the positive-class prevalence,
and ROC-AUC is below 0.5. The current model should therefore be treated
as an experimental baseline, not a production-ready churn scoring
system.

## 2. Repository contents and expected data

The project uses seven raw CSV files stored in **`data/stadiochoice/`**:

- `2_subscriber_account.csv` — subscriber/account attributes.
- `3_subscription.csv` — subscription status and timing.
- `4_viewing_behaviour.csv` — viewing sessions, duration and completion.
- `5_engagement_behaviour.csv` — engagement events, including searches.
- `5b_content_reference.csv` — content lookup used to validate viewing references.
- `6_billing_payment_history.csv` — charges, billing dates, refunds and payment events.
- `7_cancellation_retention.csv` — recorded cancellation outcomes, including `churn_indicator`.

The notebook is **`notebooks/SS3_PartB.ipynb`**. Generated figures are saved under **`notebooks/visualisations/`**, and reproducibility artefacts are saved under **`notebooks/stadiochoice_churn_outputs/`**. This README is named **`SS3_README.md`** and belongs in the **project root** (`capstone-project/`), alongside the `data/` and `notebooks/` folders.

The notebook checks source-table shapes, date fields, duplicate keys, missing identifiers and referential integrity. It keeps raw source tables unchanged and creates cleaned modelling copies.

### Key data-handling decisions

- `account` anchors the subscriber population; `household_id` is not treated as a unique subscriber key.
- Exact duplicate account rows are removed from the modelling copy.
- Category text is normalised before encoding.
- `payment_failure_count` is a repeated observation-period total, so it is **not summed across billing rows**. Event-level failures are aggregated from `payment_failure_indicator`.
- Viewing references are checked against `content`. Content titles and free-text search terms are excluded from predictors.
- The input file mapping is defined in the notebook. Keep it aligned with the seven CSVs in `data/stadiochoice/`.

## 3. Target definition and observation grain

### Observation grain

Each modelling row represents **one eligible subscriber at one month-end
index date**. A subscriber may therefore contribute multiple rows over
time. This matches the intended use case of periodically scoring
existing subscribers; it is not a test of generalisation to entirely new
subscribers.

### Target

The authoritative target is `churn_indicator` in the supplied outcome
table. A positive event means subscriber-initiated cancellation without
reactivation within 30 days, as defined by the supplied data dictionary.
The notebook uses the supplied flag rather than reconstructing the
business rule from incomplete assumptions.

For an index date (t), the target concerns the period:

**(index date, index date + 30 days\]**

A subscriber is eligible at an index date if the subscriber has been
acquired by that date and has no recorded cancellation on or before it.
The final index date is **31 May 2026**, so the full 30-day outcome
window is covered by the extract ending **30 June 2026**.

This is a forward-looking prediction target: the model must not use
information that becomes available after the index date or information
that directly reveals a later cancellation.

## 4. Data loading, validation and exploratory analysis

The notebook loads all seven tables and explicitly parses configured
timestamp fields. It then checks:

1.  table shapes against documented counts;
2.  duplicate and key behaviour at the documented grain;
3.  missing identifiers and date-field quality;
4.  referential integrity between viewing records and content;
5.  the time coverage of source records; and
6.  the distribution of recorded cancellation outcomes and the
    constructed monthly target.

The exploratory plots have different denominators and should not be
conflated:

-   **Recorded Cancellation Outcomes** counts records in the outcome
    table.
-   **Monthly Source-Record Coverage** counts timestamped viewing,
    engagement and billing source records by month. These are
    event-record counts, not unique subscribers or churn cases.
-   **30-Day Churn Prevalence by Index Date** shows the proportion of
    eligible subscriber-month rows labelled positive at each index date.
    It is a monthly panel measure, not the raw outcome-table churn
    proportion.

Source-record counts rise over time, especially for engagement. This may
reflect changing activity, coverage, or data volume; the chart alone
cannot establish which explanation is responsible. Changing source
coverage can affect engineered features and model behaviour, so it
should be investigated before operational deployment.

The prevalence plot also varies across index dates. Early zero values
and later spikes should be interpreted with their eligible subscriber
denominators and the available history in mind. A zero observed positive
proportion means no positive labels were observed in that month's
eligible panel; it does not, by itself, prove that the underlying
business had no churn.

## 5. Feature engineering

The notebook constructs a subscriber-month panel from
account/subscription attributes and activity in the **90 days ending at
the index date**. Event records are filtered to the relevant historical
window before aggregation. Future records and the final outcome table
are not predictors.

Representative engineered variables include:

  -----------------------------------------------------------------------
  Feature                             Meaning
  ----------------------------------- -----------------------------------
  `account_age_days`                  Index date minus account creation
                                      date

  `subscriber_tenure_days`            Index date minus acquisition date

  `subscription_age_days`             Index date minus subscription start
                                      date; missing if not yet started

  `view_sessions_90d`                 Distinct viewing session IDs in the
                                      previous 90 days

  `viewing_seconds_90d`               Total viewing duration in seconds
                                      over the previous 90 days

  `mean_completion_90d`               Mean recorded completion proportion
                                      over the previous 90 days

  `days_since_last_viewing`           Days from the index date to the
                                      latest prior viewing

  `engagement_events_90d`             Number of engagement event rows in
                                      the previous 90 days

  `search_events_90d`                 Count of `Search` events in the
                                      previous 90 days

  `failed_payment_events_90d`         Sum of event-level payment-failure
                                      indicators in the previous 90 days

  `amount_charged_90d`                Total charges in the previous 90
                                      days, in ZAR

  `days_since_last_billing`           Days from the index date to the
                                      latest prior billing date
  -----------------------------------------------------------------------

The notebook also engineers other billing, engagement, refund, pause and
categorical attributes where present. Consult the generated
`feature_dictionary.csv` for the fuller feature list and definitions
produced by the run.

### Leakage controls

The following are excluded from predictors: subscriber and household
identifiers, index/label-window dates, final subscription status,
subscription end date, cancellation reason, retention interactions and
future records. Dates are represented through interpretable age/tenure
or elapsed-time features where appropriate.

The panel is checked for unique `(subscriber_id, index_date)` pairs.
Event-presence indicators help distinguish no recorded activity from a
numeric zero where possible. Missingness can still carry information
about source coverage, so it should not automatically be interpreted as
customer behaviour.

## 6. Feature distributions by target

The feature-distribution figure compares selected numeric predictors
between target classes using boxplots.

-   **Subscriber tenure:** the two target groups overlap substantially.
    The plot does not show a clean separation that would allow tenure
    alone to identify churn.
-   **View sessions in the previous 90 days:** the distributions look
    very similar, suggesting this variable alone may offer limited
    separation in this sample.
-   **Failed payment events in the previous 90 days:** positive churn
    rows show a broader distribution and more failed-payment events than
    non-churn rows. This is a plausible association, but it does not
    prove that failed payments cause churn.

Boxplots summarise distributions and can hide class imbalance, unusual
values, and temporal effects. Since positive churn rows are rare,
comparisons should be read alongside class counts and the held-out
evaluation, not as proof of predictive usefulness.


## 7. Temporal Experimental Design

The data are split chronologically by `index_date`, rather than randomly by row. This simulates predicting churn on future data.

| Partition | Index-Date Period | Rows | Positive Churn Rows | Prevalence |
|:---|:---|---:|---:|---:|
| Train | Through September 2025 | 15,993 | 98 | 0.61% |
| Validation | October 2025 – January 2026 | 6,342 | 120 | 1.89% |
| Test | February – May 2026 | 6,917 | 150 | 2.17% |

### Why use a temporal split?

- **Train:** Used to fit the model on historical data.
- **Validation:** Used to tune and select the model.
- **Test:** Used to evaluate performance on a later, unseen period.

The partition dates are intended to approximate future use: train on
earlier months, select a model and threshold on the next period, then
report final performance on the later held-out period. The final test
period is not used to select the model or classification threshold.

**Important:** the same subscriber may appear in multiple partitions
because the intended task is forecasting future outcomes for the
existing subscriber base. The partitions are separate in time, but
observations are not necessarily independent at subscriber level. This
affects uncertainty estimates and is acknowledged in the notebook.

The split summary figure shows both partition sizes and positive-class
prevalence. The rising prevalence from train to validation and test
indicates that the task distribution changes over time. That shift may
contribute to performance degradation and means a single overall
prevalence should not be assumed to represent every period.

## 8. Preprocessing and categorical encoding

Preprocessing is fitted inside the modelling pipeline using training
data only:

-   Numeric predictors are median-imputed.
-   Categorical predictors are imputed with the most frequent training
    category.
-   Categorical fields are one-hot encoded.
-   Unknown categories at validation/test time are handled safely by the encoder.
-   IDs and date/label-window fields are excluded from predictors.

Keeping preprocessing inside the pipeline prevents imputation statistics
and category vocabularies from being learned using validation or test
rows. The same fitted preprocessing steps are then applied to later
periods.

## 9. Benchmarking and XGBoost model selection

The notebook compares two XGBoost configurations and a simple dummy benchmark. The dummy model is a **reference point**, not another trained tree model.

The training partition contains only **98 churn rows** and **15,895 non-churn rows**. XGBoost therefore uses `scale_pos_weight ≈ 162.194` to give positive examples more influence during training. Class weighting helps address imbalance but does not guarantee good predictions.

### Validation comparison

The primary selection metric is **Average Precision (AP)** because churn is rare.

**XGBoost regularised (selected)**
- Validation AP: **0.023780**
- Validation ROC-AUC: **0.542595**
- Validation Brier score: **0.126228**

**XGBoost shallow**
- Validation AP: **0.023437**
- Validation ROC-AUC: **0.543161**
- Validation Brier score: **0.187198**

**Dummy prevalence benchmark**
- Validation AP: **0.018921**
- Validation ROC-AUC: **0.500000**
- Validation Brier score: **0.018727**

**Why was regularised XGBoost selected?** It had the highest validation AP, the metric chosen in advance for comparing the XGBoost candidates. The shallow model's ROC-AUC was fractionally higher, but ROC-AUC was not the selection rule.

Validation churn prevalence was **0.018921 (1.89%)**. The selected model's AP of **0.023780** is above that baseline, but the improvement is small. The dummy model's constant ranking gives ROC-AUC 0.5. Its low Brier score reflects its base-rate probability predictions; it is a reminder that ranking and probability calibration measure different things.

### Classification threshold

XGBoost returns a score for each row. A threshold converts that score into a label: scores at or above the threshold are labelled churn.

The notebook tests candidate thresholds using **validation predictions only** and selects the threshold with the highest validation F1.

- **Selected threshold:** `0.5572`
- **Best validation F1:** `0.0484`

The validation F1 plot shows that F1 is low across the tested thresholds. This threshold is a validation-based statistical choice, not a threshold optimised for the financial or operational costs of retention actions.

## 10. Held-out test results

The selected model and threshold were fixed before the model was evaluated on the later test period (**February–May 2026**). These are the final held-out results.

### The results at a glance

Key metrics:

- **Average Precision:** `0.0264`
- **Test churn prevalence:** `2.17%`
- **AP lift over prevalence:** `1.218×`
- **ROC-AUC:** `0.3904`
- **Precision:** `1.66%`
- **Recall:** `10.00%`
- **F1 score:** `0.0285`
- **Brier score:** `0.1365`
- **Classification threshold:** `0.5572`


**Plain-language reading:** the model's ranking improvement over the test prevalence baseline is small. At the selected threshold it detects only 15 of the 150 actual churn rows, and most rows it flags as churn are false alarms. The ROC-AUC below 0.5 is particularly concerning because it indicates poor ordering of churn versus non-churn cases in the test period.

### Average Precision and the precision–recall curve

Test Average Precision is **0.0264**, while **2.17%** of test rows are churn cases. The AP lift is therefore about **1.218 times** the prevalence baseline — a modest improvement.

The precision–recall curve stays close to the low prevalence baseline across much of the recall range. Precision is unstable at very low recall, where only a small number of cases are being flagged. In practical terms, the model does not consistently produce a high-precision shortlist of likely churners.

AP summarises ranking quality for the rare positive class; it is **not** the percentage of predictions that are correct.

### ROC curve

Test ROC-AUC is **0.3904**, below the random-ranking reference of **0.5**. This means the model's test-period ranking is worse than random by this metric. Together with the small AP lift, this is evidence that the validation result did not generalise reliably. Do not describe the model as an effective churn predictor based on validation performance alone.


### Confusion Matrix

The matrix compares the actual outcome with the model's predicted outcome.

- **Rows:** Actual outcome
- **Columns:** Predicted outcome

| Actual Outcome | Predicted: No Churn | Predicted: Churn |
|:---|---:|---:|
| No Churn | `5,880` | `887` |
| Churn | `135` | `15` |

**Interpretation:**

- **5,880** - Correctly predicted customers who did not churn.
- **887** - False positives: customers flagged as churn risks who did not churn.
- **135** - False negatives: customers who churned but were not identified.
- **15** - True positives: customers correctly identified as churn risks.

**Key takeaway:** The model detects only 15 out of 150 actual churn cases, missing 90% of customers who churned.


The four counts have standard names:

- **True negatives (TN) = 5,880:** correctly predicted not churn.
- **False positives (FP) = 887:** predicted churn, but the row was not churn.
- **False negatives (FN) = 135:** actual churn that the model missed.
- **True positives (TP) = 15:** churn cases correctly detected.

**What does this mean operationally?**

- There were **150 actual churn rows** (`135 missed + 15 detected`).
- The model detected **15 of 150**, so recall was **10%**.
- It predicted churn for **902 rows** (`887 false alarms + 15 correct detections`).
- Only **15 of those 902 flags were correct**, so precision was **1.66%**.
- Accuracy is about **85%**, but it is misleading here because non-churn rows make up nearly all the test set.

**Bottom line:** at this threshold, the model misses 9 out of every 10 churn rows and most of its churn alerts are false positives. That is not a useful operational result without substantial improvement.

## 11. Why validation and test performance differ

Validation AP is 0.023780 and validation ROC-AUC is 0.542595 for the
selected model. On test, AP is 0.026406 but ROC-AUC falls to 0.390357.
AP is slightly higher in absolute terms, partly because test prevalence
is higher; the AP lift over prevalence remains weak. ROC-AUC falling
below 0.5 indicates a serious deterioration in ranking behaviour.

Possible contributors visible in the experiment include:

-   **Temporal distribution shift:** churn prevalence rises from 0.61%
    in training to 1.89% in validation and 2.17% in test.
-   **Sparse positive examples:** the training partition has only 98
    positive rows, limiting the patterns the model can learn.
-   **Repeated subscriber observations:** multiple monthly rows for one
    subscriber are related and do not provide as much independent
    information as the raw row count suggests.
-   **Changing source-record coverage:** viewing, engagement and billing
    event volumes change over time, which may affect feature
    distributions.
-   **Class weighting and probability quality:** weighting can help the
    model attend to rare positives but can distort probability
    calibration.

These are plausible explanations and diagnostic leads, not proven causal
findings. Additional temporal and feature-level checks are needed before
attributing the failure to one factor.

## 12. SHAP explainability

SHAP is used to inspect how the selected XGBoost model uses the
transformed predictors. A reproducible sample of up to 250 test rows is
explained to limit computation. One-hot encoded features retain their
transformed names. Global importance is measured as mean absolute SHAP
value over the explained sample; local SHAP values show contributions
for one individual prediction.

### Global SHAP feature importance

The leading features by mean absolute SHAP value are:

    Rank Feature                         Mean absolute SHAP
  ------ ----------------------------- --------------------
       1 `account_age_days`                        0.664473
       2 `subscriber_tenure_days`                  0.397196
       3 `mean_completion_90d`                     0.279887
       4 `failed_payment_events_90d`               0.239876
       5 `days_since_last_billing`                 0.225637
       6 `days_since_last_viewing`                 0.207333
       7 `amount_charged_90d`                      0.189098
       8 `viewing_seconds_90d`                     0.145438
       9 `subscription_age_days`                   0.133757
      10 `mean_pause_count_90d`                    0.125118

The SHAP beeswarm figure shows both the magnitude and direction of
feature contributions. Points to the right of zero push the model output
towards churn; points to the left push it away from churn. Colour
represents the feature value from low (blue) to high (pink/red).

Account age and subscriber tenure have the largest average absolute
contributions in this sample, followed by completion behaviour and
failed payment events. For some features, high values appear on both
sides of zero, indicating that the relationship is not simply monotonic
or may depend on interactions with other features. A global SHAP ranking
describes the fitted model's behaviour; it does **not** prove causation,
and it does not establish that the model ranks churn correctly.

Because the test ranking metrics are weak, these explanations must be
treated cautiously. SHAP explains what the model learned, including
potentially unstable or misleading patterns; it does not validate
predictive performance.

### Local SHAP explanation

The notebook also explains one sampled test observation. In the saved
run, the explained row has an actual target of **0 (not churn)** and a
predicted churn probability of approximately **0.0006**. The largest
contributions shown are negative SHAP values for `account_age_days`,
`viewing_seconds_90d`, `subscriber_tenure_days`, and
`amount_charged_90d`, pushing this observation's model output away from
churn.

This example illustrates the mechanics of a local explanation only. It
is one non-churn observation and cannot be generalised to all
subscribers or used to infer why churn occurs.

## 13. Uncertainty and probability diagnostics

### Subscriber-cluster bootstrap

The notebook estimates a 95% interval for test AP by resampling
subscribers, keeping each sampled subscriber's test rows together. The
saved run reports:

**Subscriber-cluster bootstrap AP 95% interval: 0.0161--0.0463**\
Valid bootstrap replicates: **100**

The interval is wide, reflecting substantial uncertainty. It includes
values close to and below the observed test prevalence, reinforcing that
the evidence for useful ranking is weak. The notebook uses 100
replicates, which is modest for a final inferential estimate; increase
the number of replicates for a more stable interval and report the
method and seed. The interval is a diagnostic, not a guarantee of future
performance.

### Predicted scores by actual class

The notebook's test score diagnostics show:

  Actual target      Rows   Mean predicted score   Median predicted score
  --------------- ------- ---------------------- ------------------------
  Not churn         6,767                 0.2736                   0.2477
  Churn               150                 0.2075                   0.0845

The mean and median scores for actual churn are lower than those for
actual non-churn in this test period. This is consistent with the
below-0.5 ROC-AUC and indicates that the model often assigns lower
scores to churn rows than to non-churn rows. It is a key diagnostic
finding, not merely a threshold problem: changing the threshold alone
cannot fix a poor ordering of cases.

### Brier score and MSE caution

The notebook reports a test Brier score of **0.136489** for predicted
probabilities. This is high in the context of a rare event and suggests
poor probability quality. The notebook also calculates MSE using the
**binary class predictions** (`test_pred`) rather than predicted
probabilities. Since the target is binary, this MSE is effectively the
classification error rate for 0/1 predictions, not a
probability-calibration measure. The printed value is 0.1478, which
corresponds to the share of misclassified rows at the selected
threshold. For probability calibration, use the Brier score on
`test_prob` and, ideally, reliability/calibration plots.

## 14. Figures and saved visualisations

The notebook saves figures to `notebooks/visualisations/`.
The principal figures are:

  ---------------------------------------------------------------------------------------------
  Figure                                        What it shows           Main interpretation
  --------------------------------------------- ----------------------- -----------------------
  `recorded_cancellation_outcomes.png`          Counts of outcome-table Outcome-table record
                                                records by churn        counts; not the
                                                indicator               subscriber-month model
                                                                        prevalence

  `monthly_source_record_coverage.png`          Monthly viewing,        Coverage/volume changes
                                                engagement and billing  over time; counts are
                                                record counts           not unique subscribers

  `churn_prevalence_by_index_date.png`          Positive-label          Target prevalence
                                                proportion by monthly   changes over the
                                                index date              observation period

  `feature_distributions_by_churn_target.png`   Boxplots for selected   Selected distributions
                                                features by target      overlap; failed payment
                                                                        activity appears more
                                                                        spread among positive
                                                                        rows

  `temporal_data_partition_summary.png`         Row counts and positive Chronological
                                                cases/prevalence by     evaluation and
                                                split                   increasing positive
                                                                        prevalence

  `validation_f1_by_threshold.png`              Validation F1 at        Best validation F1 is
                                                candidate thresholds    low; selected threshold
                                                                        is about 0.557

  `test_confusion_matrix.png`                   Actual versus predicted 15 true positives, 135
                                                test labels             false negatives and 887
                                                                        false positives

  `test_precision_recall_curve.png`             Test precision versus   Precision remains close
                                                recall                  to the low baseline
                                                                        over much of the curve

  `test_roc_curve.png`                          Test ROC curve against  ROC-AUC below 0.5, a
                                                random baseline         serious generalisation
                                                                        concern

  `shap_feature_importance.png`                 SHAP beeswarm summary   Shows which transformed
                                                for the selected model  features influence
                                                                        model output and in
                                                                        what direction
  ---------------------------------------------------------------------------------------------

Figures should be generated from the current executed notebook, rather
than copied from an older run. Some environments may save outputs to an
absolute local path; the notebook's `VISUALISATION_DIR` should be
configured to a repository-relative location if the project is to be
portable.

## 15. Reproducibility

### Environment

The notebook uses Python and the following main libraries:

-   `pandas` and `numpy` for data handling and feature construction;
-   `scikit-learn` for preprocessing, pipelines, baseline classification
    and evaluation metrics;
-   `xgboost` for the candidate churn classifiers;
-   `matplotlib` for figures;
-   `shap` for global and local model explanations;
-   `joblib` for saving the fitted pipeline.

The notebook prints environment information and uses a fixed random seed
in relevant modelling, sampling and bootstrap operations. Exact package
versions are not specified in this README because they should be taken
from the executed environment or a committed lock/requirements file.

### Steps to rerun

1.  Clone or open the project repository and activate the intended
    Python virtual environment.
2.  Install the notebook dependencies, including `xgboost` and `shap`.
    If `shap` is unavailable, the SHAP cells may not produce
    explanations until it is installed.
3.  Place the seven source tables in the expected data directory and
    verify the filenames in the notebook's `FILES` mapping.
4.  Set `DATA_DIR` to the correct path. Avoid machine-specific absolute
    paths in a shared repository.
5.  Open `SS3_PartB.ipynb` in Jupyter Notebook, JupyterLab or a
    compatible notebook environment.
6.  Restart the kernel and run **all cells from top to bottom**. This
    ensures that the data checks, feature panel, temporal splits,
    selected model, threshold and figures all come from one consistent
    run.
7.  Review the validation comparison and final test metrics. Do not tune
    the model or threshold against test results and then report that
    same test set as an untouched evaluation.
8.  Check the saved outputs and figures before committing or submitting
    the repository.

### Reproducibility outputs

The final notebook section writes a `stadiochoice_churn_outputs`
directory containing:

-   `experiment_metadata.json` --- experiment/environment metadata;
-   `feature_dictionary.csv` --- engineered feature definitions;
-   `selected_model_pipeline.joblib` --- fitted preprocessing and
    selected XGBoost pipeline;
-   `split_summary.csv` --- row and target counts by partition;
-   `test_metrics.csv` --- final held-out test metrics;
-   `test_predictions.csv` --- subscriber ID, index date, true target,
    predicted probability, predicted label and threshold;
-   `validation_model_comparison.csv` --- validation comparison of the
    benchmark and XGBoost candidates.

The notebook also saves visualisations separately. Treat the saved model
as a research artefact only until its performance, calibration,
stability and operational value are established. Prediction files
contain subscriber-level identifiers and should be handled as
potentially sensitive data.

### Actual project locations

```text
capstone-project/
├── data/
│   └── stadiochoice/
│       ├── account.csv
│       ├── subscription.csv
│       ├── viewing.csv
│       ├── engagement.csv
│       ├── content.csv
│       ├── billing.csv
│       └── outcome.csv
├── notebooks/
│   ├── SS3_PartB.ipynb
│   ├── visualisations/
│   └── stadiochoice_churn_outputs/
└── SS3_README.md
```

`visualisations/` contains the generated figures and SHAP importance CSV. `stadiochoice_churn_outputs/` contains the model pipeline, experiment metadata, feature dictionary, split summary, test metrics, test predictions and validation comparison. Both output folders are under `notebooks/`, not the project root.

This tree reflects the project layout described for this run. Do not commit raw or subscriber-level output data publicly unless data governance permits it.

## 16. Limitations

1.  **Weak test discrimination:** ROC-AUC is below 0.5 and AP lift over
    prevalence is small.
2.  **Severe class imbalance:** only 98 positive training rows are
    available.
3.  **Temporal drift:** positive prevalence and source-record volumes
    change across periods.
4.  **Repeated observations:** rows for the same subscriber are
    correlated across months.
5.  **Limited threshold utility:** validation F1 is low and the test
    confusion matrix contains many false positives and false negatives.
6.  **Probability quality:** the Brier score is poor; predicted scores
    should not be treated as reliable churn probabilities without
    further calibration analysis.
7.  **Explanation limitations:** SHAP describes the fitted model and
    does not establish causality or predictive validity.
8.  **Bootstrap resolution:** 100 bootstrap replicates provide only a
    preliminary uncertainty estimate.
9.  **Source coverage and missingness:** absent event records may
    reflect either real inactivity or incomplete source capture.
10. **Operational validation is absent:** no intervention-cost analysis,
    decision threshold based on business costs, prospective trial, or
    production monitoring is established by this notebook.

## 17. Recommended next steps

The immediate priority is not to deploy the current scores or to keep
changing the threshold. Investigate why the test ranking reverses:

1.  Audit the label construction and eligibility rules, especially the
    date boundaries and the mapping of outcome records to
    subscriber-month rows.
2.  Verify that all engineered features are strictly available at each
    index date and that timestamp/time-zone handling is consistent.
3.  Compare train, validation and test feature distributions,
    missingness, category frequencies, and predicted-score
    distributions.
4.  Examine whether source coverage changes or late/backfilled records
    alter historical aggregates.
5.  Evaluate whether the target definition and the intended prediction
    horizon match the practical retention decision.
6.  Use repeated temporal backtesting across several cut-off dates,
    rather than relying on a single validation/test transition.
7.  Increase subscriber-cluster bootstrap replicates and consider
    additional uncertainty diagnostics.
8.  If ranking becomes useful, assess calibration with probability-based
    metrics and reliability plots.
9.  Set a threshold using explicit business costs and intervention
    capacity, not F1 alone.
10. Keep the final test period untouched until a new experiment is fully
    specified.

## 18. Final assessment

This notebook provides a structured, reproducible XGBoost experiment
with temporal splitting, feature engineering, categorical encoding, a
benchmark, validation-based model/threshold selection, held-out
evaluation, uncertainty diagnostics and SHAP explanations. It also
identifies important issues in the data and modelling process, including
class imbalance and changing temporal prevalence.

However, the results do **not** support claiming that the current model
reliably predicts 30-day churn. The test ROC-AUC of 0.3904, AP lift of
only 1.218× prevalence, 10% recall and 1.66% precision indicate poor
practical performance at the selected threshold. The appropriate
conclusion is that the workflow is a useful modelling baseline and
diagnostic exercise, while the current XGBoost model requires
investigation and stronger temporal validation before any real retention
use.

## Link to Presentation Video
[Watch the video on YouTube](https://youtu.be/JTpA2H9cPBY)


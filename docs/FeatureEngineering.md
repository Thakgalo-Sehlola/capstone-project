# Feature Engineering

## 1. Objective

Feature engineering converts the KKBox source data into a customer-level modelling dataset for predicting the supplied `is_churn` target.

The modelling grain is **one row per labelled customer (`msno`)**.

The feature engineering process uses:

* `train.csv` for the labelled customer population;
* `user_logs` for recent listening behaviour;
* `transactions.csv` for recent transaction behaviour;
* `members_v3.csv` for customer attributes.

The supplied `is_churn` target is not reconstructed or modified.

## 2. Observation Period

The working observation period is the three calendar months immediately preceding the February 2017 churn cohort:

| Window | Period                   |
| ------ | ------------------------ |
| M3     | 2016-12-01 to 2016-12-31 |
| M2     | 2017-01-01 to 2017-01-31 |
| M1     | 2017-02-01 to 2017-02-28 |

Only information available within these periods is used for behavioural and transaction features.

March 2017 data is excluded from feature construction.

## 3. User-Log Features

The source `user_logs` data has a customer-day observation grain.

The data is aggregated from customer-day to customer-month and then to one row per labelled customer.

### 3.1 Active Days

For each monthly window:

`active_days_Mx` = number of distinct calendar dates on which the customer has a user-log record.

Unit: days.

A customer with no user-log record during the month has a value of zero.

### 3.2 Total Listening Time

For each monthly window:

`total_secs_Mx` = sum of `total_secs` across the customer's user-log records.

Unit: seconds.

### 3.3 Average Daily Listening Time

For each monthly window:

`avg_daily_secs_Mx` = `total_secs_Mx / active_days_Mx`.

The denominator is the number of observed user-log days, not the number of calendar days in the month.

For customers with zero active days, the value is treated as missing and subsequently handled during preprocessing.

### 3.4 Daily Unique-Song Observations

For each monthly window:

`total_num_unq_Mx` = sum of the daily `num_unq` values.

This is **not** a count of globally unique songs listened to during the month. It represents the sum of daily unique-song counts.

Unit: daily-unique-song observations.

### 3.5 Song-Play Event Observations

For each monthly window:

`total_song_events_Mx` is calculated as:

`num_25 + num_50 + num_75 + num_985 + num_100`

summed across the customer's user-log records.

This represents the aggregate recorded song-play counts represented by the available listening thresholds.

Unit: play-count observations.

## 4. Transaction Features

Transaction features are calculated over the same observation period:

**2016-12-01 to 2017-02-28.**

### 4.1 Transaction Count

`transaction_count` = number of transaction records for the customer during the observation period.

Unit: transactions.

### 4.2 Total Amount Paid

`total_amount_paid` = sum of `actual_amount_paid`.

Unit: KKBox's recorded payment unit.

### 4.3 Average Amount Paid

`avg_amount_paid` = `total_amount_paid / transaction_count`.

Unit: KKBox's recorded payment unit per transaction.

### 4.4 Total Listed Plan Price

`total_plan_list_price` = sum of `plan_list_price`.

Unit: KKBox's recorded price unit.

### 4.5 Average Payment Plan Duration

`avg_payment_plan_days` = mean of `payment_plan_days`.

Unit: days.

### 4.6 Auto-Renewal Rate

`auto_renew_rate` is calculated as:

`number of transactions where is_auto_renew = 1 / transaction_count`.

The denominator is the number of transaction records in the observation period.

The resulting value is a proportion between 0 and 1.

### 4.7 Cancellation Rate

`cancellation_rate` is calculated as:

`number of transactions where is_cancel = 1 / transaction_count`.

The denominator is the number of transaction records in the observation period.

The resulting value is a proportion between 0 and 1.

### 4.8 Transaction Recency

`transaction_recency_days` is:

`2017-02-28 - most recent transaction_date`.

Unit: days.

Only transaction dates within the observation period are considered.

### 4.9 Valid Expiry Count

`valid_expiry_count` = number of transactions for which `membership_expire_date >= 20000101`.

The value `19700101` is treated as an invalid/missing membership expiry date for expiry-related calculations.

The associated transaction record is retained for other transaction features because its transaction-level fields remain potentially informative.

## 5. Member Features

The following fields from `members_v3.csv` are used:

### Categorical features

* `city`
* `gender`
* `registered_via`

These are treated as categorical codes. Their numerical codes are not interpreted as continuous numerical measurements.

### Age

`bd` is treated as an age-related numerical variable.

Values outside the range **5 to 99** are treated as invalid/missing.

### Registration Date

`registration_init_time` is supplied in `YYYYMMDD` format.

The raw date is converted into:

* `registration_year`
* `registration_month`

The raw `registration_init_time` value is not supplied directly to the models.

## 6. Missing Values

Missing or unavailable monthly behavioural features are retained as missing during feature construction.

Missing numerical and categorical values are handled by the preprocessing pipeline rather than being manually replaced during feature engineering.

## 7. Output

The feature-engineering process produces one customer-level record for each labelled customer in `train.csv`.

The output contains:

* customer identifier;
* engineered behavioural features;
* engineered transaction features;
* member features;
* supplied `is_churn` target.

The generated modelling dataset is stored locally and is excluded from version control because it is derived from the private/local KKBox dataset.

## 8. Leakage Control

Feature values are calculated only from the defined observation period.

The target `is_churn` is taken directly from `train.csv`.

Information from March 2017 is excluded from feature construction.

No feature is calculated using the target variable.

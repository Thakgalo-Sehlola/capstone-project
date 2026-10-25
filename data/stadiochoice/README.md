# STADIOchoice — STADIOstream Churn: Data Extract

Seven files covering the observation period **1 January 2024 to 30 June 2026**, extracted on **1 September 2026 08:00:00**. All timestamps are **Africa/Johannesburg**, format `yyyy-MM-dd HH:mm:ss`. All money is **ZAR**; durations are **seconds**. `subscriber_id` is a consistent anonymised key across every file. **`DATA_DICTIONARY.md` carries everything section 8 asked for** — the official churn definition, categorical definitions, units, key relationships, missing-value conventions and known data issues.

## Files

### `2_subscriber_account.csv`
One row per subscriber — account, acquisition and segment.  
Rows: 2,110

| Column | Description |
|---|---|
| `subscriber_id` | String |
| `household_id` | String |
| `account_creation_date` | Datetime |
| `acquisition_date` | Datetime |
| `acquisition_channel` | Categorical |
| `acquisition_campaign` | Categorical |
| `customer_segment` | Categorical |
| `province` | Categorical |

### `3_subscription.csv`
One row per subscription — plan, fee, discount and status.  
Rows: 2,104

| Column | Description |
|---|---|
| `subscriber_id` | String |
| `subscription_id` | String |
| `subscription_plan` | Categorical |
| `subscription_start_date` | Datetime |
| `subscription_end_date` | Datetime |
| `subscription_status` | Categorical |
| `monthly_fee` | Numeric |
| `discount_amount` | Numeric |
| `promotion_code` | Categorical |

### `4_viewing_behaviour.csv`
One row per viewing session.  
Rows: 8,939

| Column | Description |
|---|---|
| `subscriber_id` | String |
| `session_id` | String |
| `viewing_timestamp` | Datetime |
| `content_id` | String |
| `content_category` | Categorical |
| `content_duration` | Integer |
| `viewing_duration` | Integer |
| `completion_rate` | Numeric |
| `device_type` | Categorical |
| `pause_count` | Integer |
| `abandonment_indicator` | Boolean |

### `5_engagement_behaviour.csv`
Event-level engagement including searches and scrolls.  
Rows: 28,118

| Column | Description |
|---|---|
| `subscriber_id` | String |
| `event_timestamp` | Datetime |
| `event_type` | Categorical |
| `search_term` | String |
| `search_result_count` | Integer |
| `content_click_indicator` | Boolean |
| `scroll_count` | Integer |

### `5b_content_reference.csv`
Content lookup (added — see README).  
Rows: 880

| Column | Description |
|---|---|
| `content_id` | String |
| `title` | String |
| `category` | Categorical |

### `6_billing_payment_history.csv`
One row per billing event.  
Rows: 32,114

| Column | Description |
|---|---|
| `subscriber_id` | String |
| `billing_date` | Datetime |
| `amount_charged` | Numeric |
| `payment_status` | Categorical |
| `payment_method` | Categorical |
| `payment_failure_indicator` | Boolean |
| `payment_failure_count` | Integer |
| `refund_indicator` | Boolean |
| `refund_amount` | Numeric |

### `7_cancellation_retention.csv`
One row per cancellation. Contains the target.  
Rows: 539

| Column | Description |
|---|---|
| `subscriber_id` | String |
| `cancellation_date` | Datetime |
| `cancellation_reason` | Categorical |
| `cancellation_channel` | Categorical |
| `churn_indicator` | Boolean |
| `reactivation_indicator` | Boolean |
| `reactivation_date` | Datetime |
| `retention_interaction_date` | Datetime |
| `retention_offer` | Categorical |
| `offer_accepted` | Boolean |
| `offer_outcome` | Categorical |

## Data quality notes

This is an extract from live operational systems and has **not** been cleaned. Expect:

- Missing values in optional / conditionally-populated fields
- Inconsistent casing and stray whitespace in some free-text and categorical fields
- A small number of duplicate records
- Occasional out-of-range or implausible values from data-capture errors
- Some date fields captured in inconsistent formats

**Target:** `churn_indicator` in `7_cancellation_retention.csv`.

**Official churn definition:** *subscription cancelled and not reactivated within 30 days*. Temporary suspensions, payment-failure lapses and accepted retention offers are **not** churn — the full treatment table is in the data dictionary. Subscribers with **no row** in file 7 never cancelled; left-join from `2_subscriber_account.csv` to build the negative class.

**Both churned and retained subscribers are included**, as required, and no outcome-based filtering has been applied.

**Historical, timestamped records — not current state.** Every behavioural, engagement and billing record carries its own timestamp, so you can set a prediction point, build features from what happened before it, and compare against the outcome after it. That was the central requirement in the request and it is what makes this extract usable.

**Leakage warning:** the whole of file 7 is post-outcome. So is `subscription_end_date` and a `subscription_status` of Cancelled or Lapsed. Cut features at `cancellation_date` for churned subscribers and at an equivalent index date for retained ones, or the model will simply learn that people who stopped watching had stopped watching.

**One addition:** `5b_content_reference.csv` was not in the request, but `content_id` in the viewing data resolved to nothing without it. It gives you title and category per content item.

_Synthetic data supplied for the CAP182 capstone project. Not real client data._
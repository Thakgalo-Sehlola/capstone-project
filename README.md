## Part A - Motivation

STADIOchoice is operating in a rapidly changing media environment as it transitions towards a streaming-first business model. The STADIOchoice briefing pack indicates that STADIOstream has approximately 7.7 million subscribers, but streaming subscribers experience substantially higher monthly churn than satellite subscribers, at 9.2% compared with 1.8%. At the same time, the cost of acquiring a streaming subscriber has increased from R240 to R355. These conditions make the retention of existing streaming subscribers increasingly important to the financial sustainability of the streaming business.

A data science project focused on subscriber churn would add value by helping STADIOchoice move from a predominantly reactive retention approach towards a more proactive, data-driven approach. The briefing pack indicates that retention activity is currently triggered when customers reach the cancellation stage, while the organisation has access to detailed behavioural information such as viewing activity, searches, pauses, scrolling and content abandonment. Combining this information with subscription, billing, cancellation and customer-support data provides an opportunity to gain deeper insight into subscriber behaviour and identify customers who may be at risk of leaving.

The project is particularly relevant to STADIOchoice's current business position because the organisation is facing rising customer acquisition costs, significant investment in content and sports rights, and strong competition within the streaming market. The briefing pack also identifies retention and the ability to identify churn early enough to act as important components of the organisation's future strategy. A churn prediction solution could therefore provide business value by supporting earlier identification of potentially at-risk subscribers, enabling more targeted retention interventions and helping STADIOchoice make better use of its available customer data.

## Part B - Problem Statement

### Context

STADIOchoice's transition towards a streaming-first business model has resulted in a growing STADIOstream subscriber base, but streaming subscribers experience significantly higher churn than satellite subscribers. The briefing pack reports a monthly streaming churn rate of 9.2%, compared with 1.8% for satellite subscribers. STADIOchoice also faces increasing subscriber acquisition costs, making the retention of existing streaming customers an important business consideration.

### Issue

STADIOchoice's current retention process is primarily reactive, with retention interventions occurring when subscribers indicate that they intend to cancel. This approach provides limited opportunity to identify subscribers who are likely to churn before reaching the cancellation stage. Although STADIOchoice has access to subscriber, subscription and behavioural data, there is a need to determine whether these data can be used to identify patterns that distinguish subscribers who are likely to churn from those who are likely to remain active.

### Relevance

The ability to identify subscribers at risk of churn before cancellation is relevant to STADIOchoice because the organisation is operating with high streaming churn, increasing acquisition costs and strong competition for streaming customers. A data-driven method for identifying churn risk could support a more proactive retention strategy and assist the business in targeting retention resources towards subscribers who are more likely to leave.

### Objectives

This project aims to develop and evaluate a machine learning model capable of predicting subscriber churn for STADIOstream. The study will analyse available subscriber and behavioural characteristics to identify factors associated with churn and determine whether these characteristics can be used to distinguish higher-risk subscribers from subscribers who are likely to remain active. Model performance will be evaluated using appropriate classification metrics to determine whether the resulting predictions are sufficiently reliable to support targeted retention decisions.


---

## Part C - Client Data Request

The proposed data request identifies the data required from STADIOchoice to support the development and evaluation of the churn prediction solution.

The detailed client data request is available in:

**[`part_c_data_request.pdf`](part_c_data_request.pdf)**

The editable source document is also retained in the repository:

**[`part_c_data_request.docx`](part_c_data_request.docx)**

The requested data includes:

- Subscriber and account information
- Subscription information
- Viewing behaviour
- Engagement behaviour
- Billing and payment history
- Cancellation and retention history
- Data definitions and metadata

The request also specifies requirements relating to historical records, timestamps, subscriber identifiers, churn definitions, missing values, relationships between datasets, data quality issues and protection of personally identifiable information.

---
## Part D - GitHub Repository Structure

The repository is organised to separate datasets, analytical code, experiments, models and project outputs.

```text
capstone-project/
│
├── data/
│   └── README.md
│
├── experiments/
│   └── setup/
│       └── README.md
│
├── models/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
├── reports/
│   └── README.md
│
├── src/
│   ├── comparison/
│   │   └── README.md
│   ├── statistical/
│   │   └── README.md
│   ├── visualisation/
│   │   └── README.md
│   └── process_logs.py
│
├── part_c_data_request.docx
├── part_c_data_request.pdf
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Part E – RAAIDD Log

The RAAIDD log below summarises the key Risks, Actions, Assumptions, Issues, Decisions, and Dependencies associated with the development of the STADIOstream subscriber churn prediction project.

### 1. Risks

- Insufficient historical churn data: The data provided by STADIOchoice may not contain enough historical examples of both churned and retained STADIOstream subscribers to train and evaluate a reliable classification model.
- Data quality and completeness: Streaming behavioural logs may contain missing, sampled, or inconsistent records, while customer-support free-text data may require substantial preprocessing.
- Class imbalance: If churned subscribers represent a substantially smaller proportion of the dataset, a model may favour the majority class and produce misleading accuracy results.
- **Data leakage:** Features may inadvertently contain information recorded after a subscriber has already begun the cancellation process, resulting in unrealistically strong model performance.

### 2. Actions

- Define the prediction target and churn observation period before modelling.
- Profile the supplied subscriber, subscription, billing, and behavioural datasets for completeness, consistency, and potential data leakage.
- Create a reproducible data-preparation and feature-engineering process.
- Establish a suitable train/validation/test strategy that respects the temporal nature of subscriber churn.
- Evaluate multiple classification models using appropriate performance metrics rather than accuracy alone.
- Analyse feature importance and model behaviour to identify characteristics associated with increased churn risk.
- Document assumptions, methodological decisions, limitations, and experimental results throughout the project lifecycle.

### 3. Assumptions

- STADIOchoice can provide sufficiently detailed historical STADIOstream subscriber and behavioural data to construct a churn prediction dataset.
- A consistent subscriber or household identifier exists across relevant streaming, subscription, and billing records.
- A reliable definition of subscriber churn can be established from cancellation and subscription records.
- Behavioural activity recorded before the prediction point can be used to construct features without exposing future information.
- The available historical data is representative enough of STADIOstream subscribers to support model development and evaluation.

### 4. Issues

- If the supplied data contains substantial missing or inconsistent subscriber identifiers, it may not be possible to reliably join viewing, subscription, and billing records.
- If behavioural logs are sampled or unavailable for important periods, some subscriber activity may be underrepresented.
- If the supplied data does not contain a sufficiently reliable churn outcome, the project may require a revised target definition or narrower scope.

### 5. Decisions

- The project will focus specifically on predicting STADIOstream subscriber churn, rather than attempting to model both streaming and satellite churn.
- Churn prediction will be framed as a classification problem, using subscriber information available before a defined prediction point to predict subsequent churn.
- Model evaluation will use classification metrics appropriate to the churn problem rather than relying on accuracy alone.
- Features containing information unavailable before the prediction point will be excluded to reduce the risk of data leakage.
- The final modelling approach will be selected based on empirical performance, interpretability, and suitability for the business objective.

### 6. Dependencies

- The project depends on STADIOchoice providing historical STADIOstream subscription, cancellation, billing, and behavioural data before meaningful modelling can begin.
- Data integration depends on consistent identifiers and compatible timestamps across the relevant data sources.
- Churn-target construction depends on having reliable cancellation or subscription-status information.
- Feature engineering depends on sufficient historical viewing and engagement records being available before each prediction period.
- Model evaluation depends on having a sufficiently large historical period containing known churn and non-churn outcomes.
- Business interpretation of the results depends on translating churn predictions and associated risk factors into a retention context.

---

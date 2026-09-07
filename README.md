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

| RAAIDD | Description |
|---|---|
| Risks | - Class imbalance: If churned subscribers make up a much smaller part of the dataset, a model may favour the larger class and produce misleading results.<br>- Data leakage: Features may unintentionally include information recorded after a subscriber has started the cancellation process, resulting in unrealistically high model performance. |
| Actions | - Establish a suitable train/validation/test strategy.<br>- Evaluate multiple classification models using relevant performance measures rather than accuracy alone.<br>- Examine important features and how the model behaves to identify factors linked to higher churn risk.<br>- Document assumptions, modelling decisions, limitations, and results throughout the project. |
| Assumptions | - STADIOchoice can provide enough historical STADIOstream subscriber and behavioural data to create a churn prediction dataset.<br>- A consistent subscriber or household identifier exists across the related streaming, subscription, and billing records.<br>- A clear definition of subscriber churn can be created from cancellation and subscription records. |
| Issues | - If behavioural logs are sampled or unavailable for important periods, some subscriber activity may not be fully represented.<br>- If the supplied data does not contain a reliable churn outcome, the project may require a changed target definition or smaller scope. |
| Decisions | - The project will focus specifically on predicting STADIOstream subscriber churn.<br>- Churn prediction will be treated as a classification task.<br>- Model evaluation will use relevant classification measures rather than accuracy alone.<br>- The final modelling approach will be selected based on actual performance, how easy the model is to understand, and its fit with the business goal. |
| Dependencies | - The project depends on STADIOchoice providing historical STADIOstream subscription, cancellation, billing, and behavioural data before modelling can begin.<br>- Data integration depends on matching identifiers and timestamps across the related data sources.<br>- Creating the churn target depends on having reliable cancellation or subscription status information.<br>- Understanding the results for the business depends on linking churn predictions and related risk factors to customer retention. |

---

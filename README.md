# Gaza Household Food-Security Prediction

[![Streamlit](https://img.shields.io/badge/app-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-classification-6E40C9)](#modeling-workflow)
[![Humanitarian Research](https://img.shields.io/badge/context-humanitarian%20research-0F766E)](#responsible-use)

> A local Streamlit research application for exploring household food-security data and comparing classification models for the `Q50` target variable.

## Project Overview

This project provides an interactive workflow for loading a household survey dataset, inspecting data quality, preprocessing mixed features, training baseline classifiers, evaluating results, and testing a single-household prediction. The interface is designed for exploratory analysis and humanitarian research support; it is not a validated hunger assessment or an operational decision system.

The application compares Logistic Regression, Random Forest, and Gradient Boosting models. It exposes accuracy, precision, recall, F1-score, confusion matrices, classification reports, feature importance, and downloadable prediction outputs.

## What the App Includes

| Area | Functionality |
|---|---|
| Data overview | Dataset shape, columns, missing values, distributions, and feature descriptions. |
| Preprocessing | Encoding of categorical values, feature preparation, and train/test splitting. |
| Model training | Comparison of Logistic Regression, Random Forest, and Gradient Boosting. |
| Evaluation | Accuracy, precision, recall, F1-score, confusion matrix, and classification report. |
| Interpretation | Tree-based feature importance and downloadable importance tables. |
| Prediction | A form for entering household attributes and viewing class probabilities. |
| Export | Processed data, predictions, reports, and serialized models where supported by the app. |

## Modeling Workflow

```text
GazaHungerData.xlsx
        ↓
Data quality review and preprocessing
        ↓
Categorical encoding + train/test split
        ↓
Logistic Regression | Random Forest | Gradient Boosting
        ↓
Metrics + confusion matrix + feature importance
        ↓
Exploratory household-level prediction
```

The current target is the survey field **`Q50`**, described in the application as a proxy for food-security or hunger severity. The target definition, label encoding, class balance, and survey design should be reviewed before interpreting any model output.

## Dataset

The included workbook is expected at:

```text
GazaHungerData.xlsx
```

The application works with approximately 1,209 household records and a broad set of survey variables covering family composition, economic conditions, displacement or living conditions, and food-access indicators. Treat these counts as dataset-specific and recompute them whenever the workbook changes.

## Run Locally

```bash
git clone https://github.com/Ahmedosrf/gaza-hunger-prediction-app.git
cd gaza-hunger-prediction-app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run gaza_hunger_app.py
```

The app should open at `http://localhost:8501`. Upload or place `GazaHungerData.xlsx` where the application expects it, then follow the tabs from data overview to preprocessing, training, evaluation, and prediction.

## Repository Structure

```text
.
├── gaza_hunger_app.py       # Streamlit application
├── requirements.txt         # Python dependencies
├── GazaHungerData.xlsx      # Survey workbook, if distributed with the project
└── README.md
```

## Responsible Use

This project uses sensitive humanitarian context and should be handled with care. Do not commit identifiable household records, phone numbers, exact addresses, or other private attributes. Predictions should not determine aid eligibility, deny services, or replace field verification. Any external deployment requires consent, documented governance, bias and subgroup evaluation, monitoring for dataset shift, and review by qualified humanitarian practitioners.

The current implementation is best described as a **research prototype**. Report the data collection process, label definition, class distribution, train/test protocol, confidence intervals, and failure cases before presenting results as evidence.

## Limitations and Next Steps

A single random train/test split can overstate performance, especially when households are correlated or the dataset is small. Future work should add stratified cross-validation, explicit leakage checks, calibrated probabilities, subgroup metrics, a versioned data dictionary, reproducible experiment configuration, and automated tests for preprocessing. The target should also be reviewed with domain experts rather than described as hunger severity without qualification.

## Maintainer

[Ahmed Osrof](https://github.com/Ahmedosrf)

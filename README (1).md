# 🦠 COVID-19 Deaths Prediction Pipeline — Dataiku DSS

> **Machine Learning pipeline for predicting COVID-19 deaths using Dataiku DSS** | Regression · Data Preparation · AutoML

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Dataiku](https://img.shields.io/badge/Dataiku-DSS-2AB1AC?style=for-the-badge&logo=dataiku&logoColor=white)
![ML](https://img.shields.io/badge/Machine%20Learning-Regression-FF6B35?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Datasets](#-datasets)
- [Pipeline Steps](#-pipeline-steps)
- [Model Performance](#-model-performance)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 🎯 Project Overview

This project implements a **complete end-to-end Machine Learning pipeline** built inside **Dataiku DSS** to predict the number of COVID-19 deaths based on epidemiological data.

The pipeline illustrates a full data science workflow — from raw data ingestion and multi-dataset joining, through cleaning and feature engineering, to model training, deployment, scoring, and evaluation.

**Key highlights:**
- 📥 Multi-source data ingestion and joining
- 🧹 Automated data preparation and feature engineering
- 🤖 AutoML regression with multiple algorithm comparison
- 📊 Model scoring on holdout test data
- 🏆 Full performance evaluation with metrics and charts

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATAIKU DSS FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [02_05_2021]──┐                                                │
│                ├──► [JOIN] ──► [Joined_covid_Data]              │
│  [Continent_   │                        │                        │
│   Country_     │              [PREPARE] │                        │
│   Mapping]─────┘                        ▼                        │
│                              [Joined_covid_Data_prepared]        │
│                                         │                        │
│                               [FILTER (US)] │                   │
│                                         ▼                        │
│                              [Filtered Dataset]                  │
│                                         │                        │
│                                [SPLIT]  │                        │
│                               ┌─────────┴──────────┐            │
│                               ▼                    ▼            │
│                        [Train_Dataset]           [Test]          │
│                               │                    │            │
│                        [TRAIN MODEL]               │            │
│                        (AutoML Regression)         │            │
│                               │                    │            │
│                        [Predict Deaths]─────► [SCORE]          │
│                                                    │            │
│                                              [Test_scored]      │
│                                                    │            │
│                                             [EVALUATE]          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Datasets

| Dataset | Type | Description |
|--------|------|-------------|
| `02_05_2021` | Source | COVID-19 worldwide statistics (confirmed cases, deaths, recoveries) as of May 2, 2021 |
| `Continent_Country_Mapping` | Source | Reference table mapping countries to continents |
| `Joined_covid_Data` | Intermediate | Merged dataset (LEFT JOIN on Country column) |
| `Joined_covid_Data_prepared` | Intermediate | Cleaned and transformed dataset |
| `Train_Dataset` | Intermediate | Training split (~70–80% of data, filtered on US) |
| `Test` | Intermediate | Test split (~20–30% of data) |
| `Test_scored` | Output | Test dataset enriched with model predictions |

---

## 🔄 Pipeline Steps

### 1. 📥 Data Import
Load both source datasets into Dataiku DSS as managed datasets.

### 2. 🔗 Dataset Join
Merge the COVID dataset with the continent mapping table using a **LEFT JOIN** on the `Country` column, ensuring no epidemiological records are lost.

### 3. 🧹 Data Preparation
Apply cleaning and transformation steps via Dataiku's Prepare recipe:
- Handle missing values (NaN imputation or removal)
- Type casting and normalization
- Drop irrelevant columns

### 4. 🔍 Filtering
Filter the dataset to retain only **United States (US)** records, reducing geographic variance and improving model precision.

### 5. 🔀 Train / Test Split
Split the prepared dataset into:
- **Train_Dataset** — used to train the model
- **Test** — held out for final evaluation

### 6. 🤖 Model Training (AutoML)
Train a **regression model** using Dataiku's AutoML Lab:
- **Target variable**: `Deaths`
- Algorithms evaluated: Ridge, Random Forest, Gradient Boosting, and more
- Optimization metrics: R², RMSE, MAE

### 7. 📊 Model Scoring
Deploy the trained model and apply it to the `Test` dataset using Dataiku's Score recipe, generating the `Test_scored` dataset with prediction columns.

### 8. 🏆 Model Evaluation
Evaluate model performance using Dataiku's Evaluate module:
- Residuals analysis
- Predicted vs. actual values chart
- Feature importance ranking

---

## 📈 Model Performance

| Metric | Description |
|--------|-------------|
| **R²** | Proportion of variance explained by the model |
| **RMSE** | Root Mean Square Error — average prediction deviation |
| **MAE** | Mean Absolute Error — robust to outliers |

> Full performance metrics and charts are available in the Dataiku DSS project Lab.

---

## 📁 Repository Structure

```
covid19-deaths-prediction-dataiku/
│
├── README.md                          # This file
├── notebook/
│   └── COVID19_Pipeline_Demo.ipynb    # Jupyter notebook demo
├── data/
│   └── sample_data.csv                # Small sample for demo purposes
└── docs/
    └── pipeline_documentation.pdf     # Full project documentation
```

---

## ▶️ How to Run

### Prerequisites
- Dataiku DSS 11+ installed
- Python 3.8+
- The two source datasets (`02_05_2021`, `Continent_Country_Mapping`)

### Steps

1. **Clone this repository**
   ```bash
   git clone https://github.com/SouheilDABBABI/covid19-deaths-prediction-dataiku.git
   cd covid19-deaths-prediction-dataiku
   ```

2. **Explore the notebook demo** (standalone, no Dataiku needed)
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   jupyter notebook notebook/COVID19_Pipeline_Demo.ipynb
   ```

3. **Replicate in Dataiku DSS**
   - Import the two datasets
   - Reproduce the Flow following the pipeline steps above
   - Run the AutoML Lab with `Deaths` as the target variable

---

## 👤 Author

**Souheil DABBABI**
*Data Scientist*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/souheil-dabbabi)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/SouheilDABBABI)

---

> *Built with Dataiku DSS · Machine Learning · Python*

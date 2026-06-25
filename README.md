# Explainable Customer Churn Analytics Using SHAP-Enhanced Ensemble Models

An end-to-end churn analytics project combining ensemble machine learning models with SHAP (SHapley Additive exPlanations) to predict customer churn and explain *why* the model makes each prediction — both at a global (overall feature importance) and individual customer level.

## 📊 Dataset

[IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features covering demographics, account information, and subscribed services.

## 🎯 Project Overview

| Phase | What's Covered |
|---|---|
| Data Cleaning | Fixed `TotalCharges` type issue, dropped invalid rows |
| EDA | 6 visualizations exploring churn drivers (contract type, tenure, charges, payment method, internet service) |
| Preprocessing | Label encoding, stratified 80/20 train/test split |
| Modeling | Logistic Regression, Random Forest, and XGBoost compared on Accuracy, Precision, Recall, F1, ROC-AUC |
| Explainability | SHAP global summary plot, individual waterfall explanations, dependence plots |
| Business Insights | Plain-language retention recommendations derived from SHAP findings |

## 🏆 Model Performance

| Model | Accuracy | Precision (Churn) | Recall (Churn) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.73 | 0.49 | 0.79 | **0.834** |
| Random Forest | 0.76 | 0.55 | 0.64 | 0.818 |
| XGBoost | 0.74 | 0.52 | 0.62 | 0.800 |

**Random Forest** was selected for SHAP analysis, as it best represents the ensemble methods central to this project while maintaining strong, competitive performance.

## 🔍 Key Findings

- **Contract type** is the single strongest churn driver — month-to-month customers churn far more than those on annual contracts
- **The first ~20 months** of a customer's tenure is the highest-risk window
- Customers **without Online Security or Tech Support** add-ons churn more
- **Fiber optic** customers churn more than DSL/no-internet customers, despite being the premium service
- **Electronic check** payment users churn more than those on automatic payment methods

## 🗂️ Repository Structure

```
├── data/
│   └── Telco_Customer_Churn_Dataset.csv
├── customer_churn_shap.ipynb    # Core deliverable: EDA, modeling, SHAP analysis
├── app.py                       # Bonus: interactive Streamlit dashboard
├── model.pkl                    # Saved trained Random Forest model
├── explainer.pkl                # Saved SHAP TreeExplainer
├── feature_columns.pkl          # Feature order used during training
├── reference_stats.pkl          # Dataset averages used in the dashboard
├── .streamlit/
│   └── config.toml              # Dark theme configuration
├── requirements.txt
└── README.md
```

## 🚀 Running the Project

### Notebook (core deliverable)
```bash
pip install -r requirements.txt
jupyter notebook customer_churn_shap.ipynb
```
Run all cells top to bottom to reproduce the full analysis.

### Interactive App (bonus add-on)
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Live demo:** [Add your deployed Streamlit Cloud link here]

> **Note:** The notebook is the primary, required deliverable for this project — it independently satisfies the full project brief (EDA, ensemble modeling, SHAP explainability). The Streamlit app is an additional interactive layer built on top of the same model and SHAP explainer, allowing live, per-customer churn risk exploration.

## 🛠️ Tech Stack

`pandas` `numpy` `scikit-learn` `xgboost` `shap` `seaborn` `matplotlib` `streamlit` `plotly`

## 📌 Author

Built as part of an internship project on explainable AI for customer analytics.

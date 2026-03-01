# 🤖 Machine Learning Toolkit — EDA, Training & Feature Engineering

> A modular, reusable Python ML framework covering **Exploratory Data Analysis**, **model training with 13+ algorithms**, **regularization**, **SMOTE**, and an interactive **Feature Selection Streamlit app** — built for rapid experimentation on any tabular dataset.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Feature%20Selection%20App-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 What This Does

A clean, modular ML codebase where each responsibility is separated into its own module — EDA, data loading, model training, and visualization. Drop in any tabular dataset and go from raw CSV to trained model in minutes.

**Two components:**

| Component | Description |
|-----------|-------------|
| `main.py` + modules | Scriptable ML pipeline — EDA → preprocessing → training → evaluation |
| `Feature Engineering_Streamlit/` | Interactive Streamlit app for feature selection using RFE, SFS, PCA, and LDA |

---

## 🗂️ Project Structure

```
ML/
│
├── main.py                          # Entry point — runs full ML pipeline on Boston dataset
├── train.py                         # 13+ models, SMOTE, regularization, evaluation
├── eda.py                           # EDA utilities — column detection, transformers, plots
├── dataframe.py                     # Dataset loaders (Boston, Startups, Churn, Mall)
├── visualize.py                     # Actual vs Predicted + Residuals plots
├── requirements.txt                 # Dependencies
│
├── data/
│   ├── boston.csv                   # Boston Housing dataset (regression)
│   ├── 50_Startups.csv              # 50 Startups dataset (regression)
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Telecom Churn (classification)
│   └── mall.csv                     # Mall Customers (clustering)
│
└── Feature Engineering_Streamlit/
    ├── FE_streamlit.py              # Streamlit app for feature selection
    ├── eda.py                       # EDA helpers used by the app
    ├── train.py                     # Training helpers used by the app
    └── requirements.txt             # App-specific dependencies
```

---

## ⚙️ Module Overview

### `eda.py` — EDA & Preprocessing Utilities

| Function | What it does |
|----------|-------------|
| `get_numerical_cols(df)` | Returns numerical column names |
| `get_categorical_cols(df)` | Returns categorical column names |
| `get_missing_values(df)` | Returns null value counts per column |
| `drop_null_values(df)` | Drops all rows with nulls |
| `del_cols(df, cols)` | Drops specified columns |
| `get_num_transformer()` | Pipeline: median imputation + StandardScaler |
| `get_cat_transformer()` | Pipeline: mode imputation + OneHotEncoder |
| `get_num_poly_transformer(degree)` | Pipeline: PolynomialFeatures + StandardScaler |
| `get_heatmap(df)` | Correlation heatmap |
| `get_pairgrid(df, features, target)` | PairGrid regression plots |
| `get_scatter_plot(df, x, y)` | Scatter plot |
| `get_boxplot(df, target, feature)` | Box plot |
| `get_countplot(df, feature, hue)` | Count plot for categorical features |

---

### `train.py` — Model Training & Evaluation

**Supported models via `get_task(task_name)`:**

| Task Name | Model |
|-----------|-------|
| `linear_regression` | LinearRegression |
| `logistic_regression` | LogisticRegression |
| `knn` | KNeighborsClassifier |
| `decision_tree` | DecisionTreeClassifier |
| `randon_forest` | RandomForestClassifier |
| `svm` | SVC |
| `naive_bayes` | BernoulliNB |
| `Ada Boost classifier` | AdaBoostClassifier (200 estimators) |
| `Ada Boost regressor` | AdaBoostRegressor (200 estimators) |
| `Gradient boost classifier` | GradientBoostingClassifier (200 estimators) |
| `Gradient boost regressor` | GradientBoostingRegressor (200 estimators) |
| `Bagging classifier` | BaggingClassifier (100 estimators) |
| `xgboost` | XGBClassifier |

**Other key functions:**

| Function | What it does |
|----------|-------------|
| `model_pipeline(X, y, num_cols, cat_cols, task, smote=0)` | Full train/test/predict pipeline |
| `model_poly_pipeline(X, y, num_cols, cat_cols)` | Polynomial regression pipeline |
| `get_resampled_smote(X, y)` | Handles class imbalance with SMOTE |
| `evaluate_model(y_test, y_pred, task)` | Returns MAPE + MSE (regression) or classification report |
| `get_lasso_mape(alpha, X, y)` | Cross-validated Lasso MAPE |
| `get_ridge_mape(alpha, X, y)` | Cross-validated Ridge MAPE |
| `get_elasticnet_mape(alpha, X, y)` | Cross-validated ElasticNet MAPE |

---

### `dataframe.py` — Dataset Loaders

| Function | Dataset | Use Case |
|----------|---------|----------|
| `get_boston()` | Boston Housing | Regression |
| `get_startup_df()` | 50 Startups | Regression |
| `get_churn_df()` | Telco Churn | Classification |

---

### `visualize.py` — Model Evaluation Plots

| Function | What it shows |
|----------|--------------|
| `plot_actual_vs_predicted(y_test, y_pred)` | Scatter of actual vs predicted with ideal line |
| `plot_residuals(y_test, y_pred)` | Residuals vs actual values — good model clusters around 0 |

---

## 🖥️ Feature Selection Streamlit App

An interactive web app where you upload **any CSV**, select your target column, and apply feature selection algorithms with live evaluation metrics.

### Feature Selection Methods Available

| Method | Description |
|--------|-------------|
| **RFE (Recursive Feature Elimination)** | Eliminates least important features iteratively using cross-validation |
| **Sequential Feature Selector (SFS)** | Greedily adds the best feature forward, step by step |
| **PCA** | Dimensionality reduction via principal components — shows cumulative variance explained |
| **LDA** | Linear Discriminant Analysis for supervised dimensionality reduction |

### Running the App

```bash
cd "Feature Engineering_Streamlit"
pip install -r requirements.txt
streamlit run FE_streamlit.py
```

### App Workflow
1. Upload any CSV dataset
2. Preview data and check column info
3. Select target column and drop irrelevant columns
4. Remove nulls if needed
5. Choose feature selection method + model
6. View selected features, evaluation metrics, and interactive plots

---

## 🚀 Quick Start (Core Pipeline)

### 1. Clone the repo
```bash
git clone https://github.com/yash0304/ml-toolkit.git
cd ml-toolkit
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the default pipeline (Boston Housing — Regression)
```bash
python main.py
```

### 4. Use in your own script
```python
import dataframe as df
import eda
import train as t
import visualize as v

# Load data
data = df.get_boston()

# Split features and target
X = df.get_feature_df(data, ['Price'])
y = df.get_target_df(data, ['Price'])

# Get column types
num_cols, cat_cols = t.get_cols(X)

# Train + predict
model, y_test, y_pred, X_train, y_train = t.model_pipeline(X, y, num_cols, cat_cols, task='linear_regression')

# Evaluate
print(t.evaluate_model(y_test, y_pred, task='regression'))

# Visualize
v.plot_actual_vs_predicted(y_test, y_pred)
v.plot_residuals(y_test, y_pred)
```

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
streamlit
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🗺️ Roadmap

- [ ] Add clustering support (K-Means, DBSCAN) for `mall.csv`
- [ ] Add model comparison dashboard in Streamlit
- [ ] Add hyperparameter tuning (GridSearchCV / Optuna)
- [ ] Add SHAP explainability plots
- [ ] Export trained models as `.pkl` files

---

## 👤 Author

**Yash Modi**
- GitHub: [@yash0304](https://github.com/yash0304)
- LinkedIn: https://www.linkedin.com/in/yash-modi-77838978/
- Role: Associate Manager — Finance & Business Transformation, HCL Technologies
- MBA: Data Science & Operations Management, IFMR GSB (Rank 6)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
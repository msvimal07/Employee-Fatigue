# Employee Fatigue & Productivity Prediction System

An enterprise-grade workforce analytics and fatigue prediction platform powered by **AutoGluon** and built with **Streamlit**.

## Project Overview
This project simulates, analyzes, and predicts employee fatigue levels (`Low`, `Medium`, `High`) based on a comprehensive set of workplace variables, rest metrics, health indicators, and productivity outputs. It enables proactive management of employee burnout and task allocation.

---

## Workspace Structure
* **`app.py`**: A premium, interactive Streamlit web application for single fatigue evaluation, workforce analytics, and bulk prediction processing.
* **`Model_Training_AutoGluon.ipynb`**: Machine learning training pipeline using AutoGluon with sequential Random Search hyperparameter tuning.
* **`EDA.ipynb`**: Extensive Exploratory Data Analysis of the dataset, distributions, correlations, and feature importances.
* **`dataset.py`**: Synthetic data generator script simulating realistic workspace dependencies, missing values, and outliers.
* **`employee_fatigue_productivity_dataset.csv`**: The 50,000-record employee workday dataset.
* **`models/`**: Compact, optimized best-performing model artifacts and training metadata.

---

## Getting Started

### 1. Install Dependencies
Ensure you have the required libraries installed:
```bash
pip install autogluon streamlit pandas numpy plotly scikit-learn matplotlib seaborn
```

### 2. Run the Streamlit Application
Launch the web interface from the project root:
```bash
streamlit run app.py
```

### 3. Re-train the Model
Open and execute the cells in `Model_Training_AutoGluon.ipynb` to re-run baseline evaluations and hyperparameter search.

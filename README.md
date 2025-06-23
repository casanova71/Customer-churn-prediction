# 📉 Telco Customer Churn Prediction

This project predicts customer churn for a telecom company using machine learning. It covers a full ML lifecycle: EDA, preprocessing, model training with multiple algorithms (Random Forest, CatBoost, XGBoost), hyperparameter tuning with Optuna, model interpretability prep with SHAP, and deployment via Streamlit.
THIS IS MY ONENOTE LINK WHERE I HAVE EXPLAINED EACH LINE OF THE CODE - https://1drv.ms/o/c/0c0717e7292cddce/Es7dLCnnFwcggAzPLAAAAAABgSID-L4g-LkgtygDrhAhAQ?e=azOCQS
---

## 🚀 Project Overview

- Dataset: Telco Customer Churn (from Kaggle)
- Goal: Predict whether a customer will churn (Yes/No)
- Target Variable: Churn
- Models Used:
  - Random Forest Classifier
  - CatBoost Classifier
  - XGBoost Classifier (with Optuna tuning)
- Interface: Streamlit Web App
- Tools: pandas, scikit-learn, CatBoost, XGBoost, Optuna, SHAP, Streamlit, joblib

---

## 📂 Project Structure

customer-churn-prediction/
│
├── data/
│ ├── Telco-Customer-Churn.csv
│ └── telco_cleaned.csv
│
├── models/
│ ├── churn_xgb_pipeline.pkl
│
├── notebooks/
│ ├── 01_eda_and_cleaning.ipynb
│ └── 02_feature_engineering_and_modeling.ipynb
│
├── app/
│ └── churn_app.py
│
├── requirements.txt
└── README.md

---

## 🧪 ML Workflow

### ✅ Phase 1: Data Cleaning + Exploration

- Cleaned and transformed TotalCharges column
- Dropped irrelevant columns (e.g., customerID)
- Encoded categorical variables
- Checked class imbalance
- Separated target (Churn) and features

### ✅ Phase 2: Baseline Modeling

- Built preprocessing pipeline (OneHotEncoder + StandardScaler)
- Trained baseline models:
  - 🎯 RandomForestClassifier
  - 🎯 CatBoostClassifier
- Evaluated using classification_report and F1-score
- Visualized feature importance (Random Forest)

### ✅ Phase 3: Hyperparameter Tuning (Optuna + XGBoost)

- Used Optuna with StratifiedKFold CV to optimize:
  - learning_rate, max_depth, l2_leaf_reg, random_strength, etc.
- Trained final XGBoostClassifier on full dataset with best parameters
- Saved final pipeline (preprocessor + model) with joblib

```python
joblib.dump(final_model, "models/churn_xgb_pipeline.pkl")

🖥️ Streamlit Web App
Built a UI using Streamlit to interactively predict churn

User inputs features in a form

Predicts probability of churn and displays result

bash
Copy code
streamlit run churn_app.py
Output example:

yaml
Copy code
Churn Probability: 83.2%
Predicted Outcome: Yes (Churn)
🔍 Future Enhancements
Integrate SHAP for local/global interpretability

Deploy via FastAPI for production use

Store user predictions in cloud database

Add LSTM or Transformer models for sequence churn prediction (monthly behavior)

💻 Tools & Libraries
Tool	Role
pandas	Data wrangling
scikit-learn	Preprocessing, metrics
CatBoost	Boosting-based classifier
XGBoost	Final model, optimized with Optuna
RandomForest	Baseline tree-based model
Optuna	Hyperparameter tuning
SHAP	Explainable ML (planned)
Streamlit	Web application UI
joblib	Model serialization

📜 Run It Locally
Clone the repo:

bash
Copy code
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
Install requirements:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run churn_app.py
🙋‍♂️ Author
👨‍💻 Archit Patil

🧠 Aspiring Machine Learning Engineer

🎮 AI + God of War Reels on Instagram

📍 Passionate about building real-world ML apps

⭐️ Support
If you found this helpful:

🌟 Star the repo

🍴 Fork it

🤝 Share with others

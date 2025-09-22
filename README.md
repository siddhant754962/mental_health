

# 🧠 Mental Health Prediction using Random Forest

## 📌 Project Overview

This project uses **machine learning (Random Forest Classifier)** to predict whether an individual is at risk of requiring **mental health treatment** based on workplace and personal factors.
It integrates **data preprocessing, EDA, model building, evaluation, and deployment** via a **Streamlit web app**.

---

## 🎯 Problem Definition

Mental health issues are often under-reported in workplaces. This project aims to build a **predictive system** that:

* Analyzes employee-related features (age, gender, family history, work environment, etc.)
* Predicts whether they are likely to **seek treatment**
* Provides **risk scoring, interpretability, and “what-if” scenarios** for better awareness

---

## 📂 Dataset

* **Source**: [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
* **Size**: \~1,250 rows × 27 columns
* **Target Variable**: `treatment` (Yes/No → 1/0)
* **Key Features**:

  * Demographics: `Age`, `Gender`, `Country`
  * Workplace: `benefits`, `care_options`, `coworkers`, `anonymity`
  * Family & History: `family_history`, `mental_health_consequence`
  * Others: `remote_work`, `seek_help`, etc.

---

## 🔍 Exploratory Data Analysis (EDA)

* **Missing values handled** (imputation/encoding)
* **Categorical features encoded** using `OneHotEncoding`
* **Pair plots & correlation heatmaps** to explore feature-target relationships
* **Balance Check**:

  * Treatment (Yes) → 50.6%
  * Treatment (No) → 49.4%

---

## ⚙️ Data Preprocessing

1. Handled missing values
2. Encoded categorical features into numeric
3. Standardized the dataset
4. Ensured all input features align with model expectations

---

## 🤖 Model Training

* **Algorithm**: Random Forest Classifier
* **Split**: Train/Test (80/20)
* **Evaluation Metrics**: Accuracy, Recall, Precision, F1 Score, ROC-AUC

📊 **Final Model Performance**:

* **Accuracy**: 80.9%
* **Recall**: 85.3%
* **F1 Score**: 81.3%
* **ROC-AUC**: 0.89
* **Confusion Matrix**:

  ```
  [[99, 30],
   [18, 105]]
  ```

✅ The model balances **accuracy and recall**, ensuring fewer false negatives (important for mental health detection).

---

## 💾 Model Saving

We save both the trained model and feature columns for deployment:

```python
import joblib
joblib.dump(rf_model, "mental_health_rf_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
```

---

## 🌐 Deployment – Streamlit App

The project includes a **modern, step-based Streamlit application** with:
✔️ Custom CSS for a clean UI
✔️ Step-by-step feature input
✔️ Risk scoring & explanations
✔️ “What-if” analysis to see how changing one factor impacts risk
✔️ Fallback **dummy model** (so app runs even without trained model file)

### Run the App

```bash
streamlit run app.py
```

---

## 📊 App Preview

* **Step 1:** Enter demographics (Age, Gender, Country)
* **Step 2:** Enter family history & workplace support
* **Step 3:** Model gives prediction:

  * 🟢 **Low Risk** → May not need treatment
  * 🔴 **High Risk** → Likely needs treatment
* **Extra:**

  * Risk Score (%)
  * Prediction Explanation
  * What-if Scenarios

---

## 🚀 Tech Stack

* **Python** (Pandas, NumPy, Scikit-learn, Joblib)
* **Visualization**: Matplotlib, Seaborn
* **ML Model**: Random Forest Classifier
* **Deployment**: Streamlit

---

## 📌 Future Improvements

* Integrate **explainability (SHAP, LIME)** for deeper insights
* Expand to **multi-class mental health conditions** (not just treatment vs. no treatment)
* Build a **dashboard for organizations** to analyze workplace trends

---

## 👨‍💻 Author

**Siddhant Patel**

* 🔗 GitHub: [@siddhant754962](https://github.com/siddhant754962)
* 📧 Email: [kumarsidhant144@gmail.com](mailto:kumarsidhant144@gmail.com)

---

✨ *This project highlights how AI can help organizations better understand and support mental health in the workplace.*



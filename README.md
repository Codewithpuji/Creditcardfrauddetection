# 💳 Credit Card Fraud Detection using Machine Learning
**MSc Data Analytics Dissertation — Poojitha Kalyanam (2024)**

Benchmarking five machine learning algorithms on a real-world imbalanced financial dataset to detect fraudulent credit card transactions. The study applies CRISP-DM methodology and addresses the core challenge of class imbalance in fraud detection.

---

## 📌 Problem Statement

Credit card fraud costs Europe alone up to **€1.5 billion annually**. Detecting fraud is a highly imbalanced classification problem — fraudulent transactions represent only **0.17%** of all records. Standard accuracy metrics are misleading on such data; this study evaluates models using precision, recall, F1-score, and ROC-AUC across both balanced and unbalanced datasets.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Fraudulent | 492 (0.17%) |
| Features | 30 (V1–V28 PCA-transformed + Amount + Time) |
| Period | September 2013, European cardholders |

---

## ⚙️ Methodology (CRISP-DM)

```
Business Understanding → Data Understanding → Data Preparation → Modelling → Evaluation → Deployment
```

### Preprocessing
- **RobustScaler** applied to `Amount` feature (outlier-tolerant)
- **Time** feature normalised to [0, 1]
- **SMOTE** (Synthetic Minority Oversampling Technique) used to balance the training set
- **PCA** transformation already applied to features V1–V28 (privacy-preserving)
- Dataset split: Training (240,000) | Testing (22,000) | Validation (26,000)

---

## 🤖 Models Compared

| Model | Balanced Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score |
|---|---|---|---|---|
| **Random Forest** ⭐ | **0.96** | **0.98** | **0.93** | **0.96** |
| Logistic Regression | 0.94 | 0.93 | 0.96 | 0.94 |
| Linear SVC | 0.94 | 0.93 | 0.96 | 0.94 |
| Gradient Boosting | 0.93 | 0.92 | 0.94 | 0.93 |
| Shallow Neural Network | 0.91 | 0.89 | 0.93 | 0.91 |

> All metrics evaluated on **balanced data**. Unbalanced data results in artificially inflated accuracy due to class dominance.

---

## 🏆 Key Findings

**Random Forest** identified as the optimal production model:
- Highest precision **(0.98)** — fewest false positives
- Strong recall **(0.93)** — detects most actual fraud cases
- Best F1-score **(0.96)** — strongest balance of precision and recall
- Most robust to overfitting via ensemble averaging of decision trees
- Handles class imbalance better than single-model approaches

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

**Libraries:** `scikit-learn` · `TensorFlow/Keras` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `imbalanced-learn (SMOTE)`

**Models:** Random Forest · Gradient Boosting · Shallow Neural Network · Linear SVC · Logistic Regression

**Techniques:** PCA · SMOTE · RobustScaler · Confusion Matrix · ROC-AUC · Cross-Validation

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── credit_card_fraud_detection.ipynb   # Full notebook (EDA + preprocessing + modelling)
├── requirements.txt                    # Dependencies
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Codewithpuji/Creditcardfrauddetection.git
cd Creditcardfrauddetection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project root.

**4. Run the notebook**
```bash
jupyter notebook credit_card_fraud_detection.ipynb
```

Or open directly in **[Google Colab](https://colab.research.google.com/drive/16EPBee4FmvsI5pvG2T13UUgJOwkZhYbT?usp=sharing)**

---

## 📈 Results Visualisation

The notebook includes:
- Class distribution (fraud vs. non-fraud)
- Correlation matrix heatmap
- Feature distribution histograms
- Confusion matrices for all 5 models
- ROC curves for all 5 models
- Model accuracy comparison table

---

## 📚 Research Context

This dissertation was completed as part of the **MSc Data Analytics** programme at the **University for the Creative Arts, Germany (2024)**. It follows CRISP-DM methodology and addresses documented gaps in existing literature around systematic balanced vs. unbalanced comparison and computational efficiency analysis.

---

## 👩‍💻 Author

**Poojitha Kalyanam** — Data Analyst | MSc Data Analytics  
📍 Berlin, Germany  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/poojitha-kalyanam)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Codewithpuji)

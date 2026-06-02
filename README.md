# 💳 Credit Card Fraud Detection using Machine Learning
**MSc Data Analytics Dissertation — Poojitha Kalyanam (2024)**

Benchmarking five machine learning algorithms on a real-world imbalanced financial dataset to detect fraudulent credit card transactions, following the CRISP-DM methodology.

---

## 📌 Problem Statement

Credit card fraud costs Europe alone up to **€1.5 billion annually**. Detecting fraud is a highly imbalanced classification problem — fraudulent transactions represent only **0.17%** of all records. Standard accuracy metrics are misleading on such data; this study evaluates all models using precision, recall, F1-score, and ROC-AUC across both unbalanced and balanced datasets.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total transactions | 284,807 |
| Fraudulent | 492 (0.17%) |
| Features | 30 (V1–V28 PCA-transformed + Amount + Time) |
| Period | September 2013, European cardholders |

---

## ⚙️ Methodology (CRISP-DM)

```
Business Understanding → Data Understanding → Data Preparation → Modelling → Evaluation → Deployment
```

### Preprocessing
- **RobustScaler** applied to `Amount` feature (outlier-tolerant scaling)
- **Time** feature normalised to [0, 1]
- **Random undersampling** of majority class to balance training data (492 fraud + 492 non-fraud)
- Dataset split — Unbalanced: Train (240,000) | Test (22,000) | Val (22,807)
- Dataset split — Balanced: Train (700) | Test (142) | Val (142)

---

## 🤖 Models Compared

| Model | Balanced Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score |
|---|---|---|---|---|
| **Random Forest** ⭐ | **0.96** | **0.98** | **0.93** | **0.96** |
| Logistic Regression | 0.94 | 0.93 | 0.96 | 0.94 |
| Linear SVC | 0.94 | 0.93 | 0.96 | 0.94 |
| Gradient Boosting | 0.93 | 0.92 | 0.94 | 0.93 |
| Shallow Neural Network | 0.91 | 0.89 | 0.93 | 0.91 |

> All metrics evaluated on **balanced data**. Unbalanced results show artificially high accuracy due to class dominance and are reported separately in the notebook.

---

## 🏆 Key Finding

**Random Forest** identified as the optimal production model:
- Highest precision **(0.98)** — fewest false positives
- Strong recall **(0.93)** — catches most real fraud cases
- Best F1-score **(0.96)** — strongest balance of precision and recall
- Most robust to overfitting via ensemble averaging of decision trees

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

**Libraries:** `scikit-learn` · `tensorflow` · `pandas` · `matplotlib` · `seaborn`

**Models:** Random Forest · Gradient Boosting · Shallow Neural Network · Linear SVC · Logistic Regression

**Techniques:** RobustScaler · Random Undersampling · ROC-AUC · Confusion Matrix · Cross-Validation

---

## 📁 Repository Structure

```
Creditcardfrauddetection/
│
├── creditcardfrauddetection.py   # Full script (EDA + preprocessing + all 5 models)
├── requirements.txt              # Dependencies
└── README.md
```

---

## 🚀 How to Run

### Option A — Google Colab (Recommended)
Open directly: **[Open in Colab](https://colab.research.google.com/drive/16EPBee4FmvsI5pvG2T13UUgJOwkZhYbT?usp=sharing)**

You'll need a `kaggle.json` API key file — download it from your [Kaggle account settings](https://www.kaggle.com/settings).

### Option B — Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Codewithpuji/Creditcardfrauddetection.git
cd Creditcardfrauddetection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset manually**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root. Then remove the Kaggle shell commands at the top of the script before running.

**4. Run**
```bash
python creditcardfrauddetection.py
```

---

## 📈 Visualisations Included

- Class distribution bar chart (fraud vs. non-fraud)
- Correlation matrix heatmap (all 30 features)
- Feature distribution histograms
- Confusion matrices for all 5 models (balanced + unbalanced)
- ROC curves for all 5 models

---

## 📚 Research Context

MSc Data Analytics dissertation, **University for the Creative Arts, Germany (2024)**. Follows CRISP-DM methodology with a systematic comparison of models on both balanced and unbalanced data — addressing a documented gap in existing fraud detection literature.

---

## 👩‍💻 Author

**Poojitha Kalyanam** — Data Analyst | MSc Data Analytics
📍 Berlin, Germany
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/poojitha-kalyanam)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Codewithpuji)

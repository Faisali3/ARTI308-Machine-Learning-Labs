# 🌲 Decision Trees & Random Forests

> A hands-on machine learning lab exploring tree-based classification models using two real-world datasets — **Kyphosis** (medical) and **LendingClub Loans** (financial).

---

## Decision Trees & Random Forests (Concepts)

**Dataset:** `kyphosis.csv`  
**Goal:** Predict whether kyphosis (a spinal deformity) is present or absent after corrective surgery.

### 🔬 Dataset Overview

| Column   | Description                                              |
|----------|----------------------------------------------------------|
| Kyphosis | Target — `absent` or `present` after surgery            |
| Age      | Patient age in months                                    |
| Number   | Number of vertebrae involved in the operation            |
| Start    | Index of the topmost vertebra operated on                |

**81 patients**, **4 columns**, **no missing values**.

### 🔁 Workflow

1. **EDA** — Seaborn pairplot colored by Kyphosis outcome
2. **Train/Test Split** — 70% train, 30% test
3. **Decision Tree Classifier** — single tree, no depth limit
4. **Random Forest Classifier** — 100 estimators (ensemble of trees)
5. **Evaluation** — confusion matrix + classification report

### 📊 Results

| Model           | Accuracy | Notes                              |
|-----------------|----------|------------------------------------|
| Decision Tree   | 76%      | Some overfitting on minority class |
| Random Forest   | 80%      | Better overall accuracy            |

**Key takeaway:** Random Forest reduces the variance of a single decision tree and improves generalization, particularly when data is limited (n=81).

---

## Random Forest Project (LendingClub)

**Dataset:** `loan_data.csv`  
**Goal:** Predict whether a borrower will **fully repay** their loan using LendingClub data from 2007–2010.

### 🏦 Dataset Overview

**9,578 loans**, **14 features**, no missing values.

| Column              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `credit.policy`     | 1 if borrower meets LendingClub's underwriting criteria, else 0             |
| `purpose`           | Loan purpose (credit card, debt consolidation, small business, etc.)        |
| `int.rate`          | Interest rate (higher = riskier borrower)                                   |
| `installment`       | Monthly installment amount                                                  |
| `log.annual.inc`    | Log of self-reported annual income                                          |
| `dti`               | Debt-to-income ratio                                                        |
| `fico`              | FICO credit score                                                           |
| `days.with.cr.line` | Number of days borrower has had a credit line                               |
| `revol.bal`         | Revolving balance (unpaid at end of billing cycle)                          |
| `revol.util`        | Revolving line utilization rate                                             |
| `inq.last.6mths`    | Number of creditor inquiries in last 6 months                               |
| `delinq.2yrs`       | Times 30+ days past due in past 2 years                                     |
| `pub.rec`           | Number of derogatory public records (bankruptcy, liens, judgments)          |
| `not.fully.paid`    | **Target** — 1 if loan was NOT fully paid back, 0 if it was                |

### 🔁 Workflow

1. **EDA**
   - FICO score histograms split by `credit.policy` and `not.fully.paid`
   - Countplot of loan purposes colored by repayment outcome
   - Joint plot of FICO score vs. interest rate
   - Linear regression plots (`lmplot`) separated by `not.fully.paid` and `credit.policy`

2. **Feature Engineering** — `purpose` column converted to dummy variables using `pd.get_dummies`

3. **Train/Test Split** — 70% train, 30% test

4. **Decision Tree Classifier** — baseline single-tree model

5. **Random Forest Classifier** — 600 estimators for robust ensemble

6. **Evaluation** — confusion matrix + classification report

### 📊 Results

| Model           | Accuracy | Class 0 (Paid) F1 | Class 1 (Not Paid) F1 | Notes                                  |
|-----------------|----------|--------------------|-----------------------|----------------------------------------|
| Decision Tree   | 74%      | 0.84               | 0.21                  | Decent but unstable on minority class  |
| Random Forest   | 84%      | 0.91               | 0.04                  | High overall accuracy, class imbalance effect |

**Key takeaway:** Both models struggle with the minority class (`not.fully.paid` ≈ 16% of data), a classic **class imbalance** problem. Random Forest achieves higher overall accuracy by predicting the majority class more reliably. Techniques like SMOTE or class weighting could improve recall on defaulted loans.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebooks

```bash
jupyter notebook
```

Then open either notebook from the browser UI.

---

## 🧠 Concepts Covered

| Concept                    | Description                                                               |
|----------------------------|---------------------------------------------------------------------------|
| **Decision Tree**          | A tree-based model that splits data based on feature thresholds           |
| **Random Forest**          | An ensemble of decision trees using bagging + random feature subsets      |
| **Overfitting**            | When a model learns noise; DT is prone to this without pruning            |
| **Class Imbalance**        | When target classes are unequal — affects minority class recall           |
| **Dummy Variables**        | Converting categorical features to binary columns for ML compatibility    |
| **Confusion Matrix**       | Table showing true/false positives and negatives                          |
| **Classification Report**  | Precision, recall, F1-score per class                                     |

---

## 📦 Dependencies

| Library      | Version  | Purpose                         |
|--------------|----------|---------------------------------|
| pandas       | ≥ 1.3    | Data loading and manipulation   |
| numpy        | ≥ 1.21   | Numerical operations            |
| matplotlib   | ≥ 3.4    | Base plotting                   |
| seaborn      | ≥ 0.11   | Statistical visualization       |
| scikit-learn | ≥ 0.24   | ML models and evaluation        |

---

## 📝 Notes

- The `kyphosis.csv` dataset is a classic small medical dataset (81 rows) — good for understanding model behavior with limited data.
- The `loan_data.csv` is a cleaned subset of LendingClub public data (pre-IPO, 2007–2010) — representing a real-world imbalanced classification problem.
- Tree visualization code using `pydot`/`graphviz` is included as comments in Notebook 1; install those separately if you want to render the tree structure.

---

## 🏫 Course

**ARTI308 — Machine Learning**

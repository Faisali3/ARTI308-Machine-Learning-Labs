# K-Nearest Neighbors (KNN) Lab

## Overview

This lab applies the **K-Nearest Neighbors (KNN)** classification algorithm to an anonymized real-world dataset. The goal is to build a model that predicts a binary **TARGET CLASS** from a set of numerical features, using the full machine learning pipeline: exploration → preprocessing → training → evaluation → optimization.

---

## Files

| File | Description |
|------|-------------|
| `02-K_Nearest_Neighbors_Assignment_COMPLETED.ipynb` | ✅ Completed assignment notebook with all code and outputs |
| `01-K_Nearest_Neighbors.ipynb` | Reference lecture notebook |
| `KNN_Project_Data.csv` | Project dataset (1,000 samples, 10 features + target) |
| `Classified_Data.csv` | Lecture dataset used in the walkthrough |

---

## Dataset

- **Source:** `KNN_Project_Data.csv`
- **Samples:** 1,000
- **Features:** 10 anonymized numerical columns (`XVPM`, `GWYH`, `TRAT`, `TLLZ`, `IGGA`, `HYKR`, `EDFS`, `GUUB`, `MGJM`, `JHZC`)
- **Target:** `TARGET CLASS` — binary classification (0 or 1)

---

## Lab Steps

### 1. Import Libraries
Standard data science stack: `pandas`, `numpy`, `seaborn`, `matplotlib`, and `scikit-learn`.

### 2. Exploratory Data Analysis (EDA)
- Load and inspect the dataset
- Generate a **pairplot** (colored by `TARGET CLASS`) to visualize feature distributions and class separability

### 3. Feature Scaling
KNN is distance-based, so scale matters. Applied **`StandardScaler`** to normalize all features to zero mean and unit variance before training.

### 4. Train/Test Split
- Split: **70% training / 30% testing** (`random_state=42` for reproducibility)

### 5. KNN with K=1
- Trained a `KNeighborsClassifier` with `n_neighbors=1`
- Evaluated using a **confusion matrix** and **classification report**

### 6. Elbow Method — Choosing the Best K
- Looped over K values from 1 to 39
- Computed **error rate** for each K
- Plotted the **Error Rate vs. K Value** curve to find the optimal K

### 7. Retrain with Optimal K (K=23)
- Retrained the model using K=23 (identified as optimal from the elbow plot)
- Compared performance between K=1 and K=23 using classification metrics

---

## Key Results

| Metric | K=1 | K=23 |
|--------|-----|------|
| Overall Accuracy | ~0.82 | ~0.94 |
| Model Complexity | High (overfitting risk) | Balanced |

> **Takeaway:** K=1 memorizes training data and overfits. Increasing K to ~23 smooths the decision boundary and significantly improves generalization performance.

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn jupyter

# Launch Jupyter
jupyter notebook 02-K_Nearest_Neighbors_Assignment_COMPLETED.ipynb
```

Make sure `KNN_Project_Data.csv` is in the same directory as the notebook.

---

## Concepts Covered

- K-Nearest Neighbors classification
- Feature standardization / normalization
- Train/test split
- Confusion matrix & classification report (precision, recall, F1)
- Elbow method for hyperparameter tuning
- Bias-variance tradeoff in KNN

---

## Course

**ARTI308 – Machine Learning**

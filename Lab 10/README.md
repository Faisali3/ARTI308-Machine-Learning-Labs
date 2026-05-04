# Support Vector Machines (SVM) Lab — ARTI308 Machine Learning

## Overview

This lab introduces **Support Vector Machines (SVM)**, a supervised learning algorithm used for classification and regression. It consists of two notebooks:

| File | Description |
|------|-------------|
| `01-Support_Vector_Machines.ipynb` | Lecture notebook — SVM on the Breast Cancer dataset |
| `02-SVM_Assignment_Completed.ipynb` | Assignment — SVM on the Iris flower dataset |

---

## Notebooks

### 01 — Lecture: Breast Cancer Classification
Demonstrates SVM on sklearn's built-in **Breast Cancer** dataset (569 samples, 30 features).

- Loads and explores the dataset
- Trains a base `SVC()` model
- Evaluates with confusion matrix and classification report
- Tunes hyperparameters using **GridSearchCV** (`C`, `gamma`, `kernel`)

### 02 — Assignment: Iris Flower Classification
Applies SVM to the classic **Iris** dataset (150 samples, 4 features, 3 species).

- **EDA**: Pairplot and KDE visualization
- **Train/Test Split**: 70/30, `random_state=101`
- **Base SVC model**: ~98% accuracy out of the box
- **GridSearchCV**: Searches over `C` and `gamma` values with RBF kernel
- **Conclusion**: *Iris Setosa* is the most separable species

---

## Dataset

The Iris dataset is loaded directly from `sklearn.datasets.load_iris()`, which provides:
- 150 samples across 3 species: *setosa*, *versicolor*, *virginica*
- Features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (all in cm)

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Key Results (Assignment)

| Model | Accuracy |
|-------|----------|
| Base SVC | ~98% |
| GridSearchCV SVC (C=1, gamma=0.1, kernel=rbf) | ~98% |

The model performs exceptionally well on Iris — the dataset is small and clean, so tuning yields marginal gains.

---

## Concepts Covered

- **Support Vector Classifier (SVC)** — finds the optimal hyperplane separating classes
- **RBF Kernel** — maps data into higher dimensions for non-linear separation
- **GridSearchCV** — exhaustive hyperparameter search with cross-validation
- **Confusion Matrix & Classification Report** — evaluating multi-class classifiers


# ARTI308 – Lab 6
## Linear Regression with Ecommerce Customers Dataset

This project applies **Linear Regression** to the **Ecommerce Customers** dataset in order to predict how much a customer spends yearly based on their activity on the platform.

The notebook follows a standard machine learning workflow including data exploration, preprocessing, model training, and evaluation.

---

# 1. Load the Dataset

The dataset `Ecommerce Customers.csv` is loaded using **Pandas**.

It contains customer information such as:

- Avg. Session Length
- Time on App
- Time on Website
- Length of Membership
- Yearly Amount Spent

Example:

```python
import pandas as pd
df = pd.read_csv("Ecommerce Customers.csv")
```

---

# 2. Data Exploration

Basic exploration is performed to understand the dataset structure.

Methods used:

- `df.head()` – preview first rows
- `df.info()` – check data types and missing values
- `df.describe()` – statistical summary

Findings:

- Dataset contains **500 rows**
- No missing values
- Features are already numeric and suitable for modeling

---

# 3. Feature Selection

The following features were used for prediction:

- Avg. Session Length
- Time on App
- Time on Website
- Length of Membership

Target variable:

- **Yearly Amount Spent**

Example:

```python
X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']
```

---

# 4. Train/Test Split

The dataset is divided into:

- **70% training data**
- **30% testing data**

This allows the model to be trained and then evaluated on unseen data.

Example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

---

# 5. Model Training

A **Linear Regression** model from `sklearn` is used.

Example:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

---

# 6. Predictions

After training the model, predictions are generated using the test dataset.

```python
predictions = model.predict(X_test)
```

---

# 7. Model Evaluation

To measure model performance, the following metrics are calculated:

- **MAE (Mean Absolute Error)**
- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**

Example:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
```

Lower values indicate better prediction accuracy.

---

# 8. Visualization

A scatter plot is used to compare **Actual vs Predicted spending**.

This helps visually check how well the model predictions match the real values.

Example:

```python
plt.scatter(y_test, predictions)
plt.xlabel("Actual Spending")
plt.ylabel("Predicted Spending")
```

---

# 9. Model Coefficients

The coefficients show how much each feature affects yearly spending.

Example:

```python
coefficients = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
```

Features with larger coefficients have a stronger impact on spending.

---

# Conclusion

This lab demonstrates how **Linear Regression** can be used to predict customer spending using behavioral features.

Key insights:

- **Length of Membership** tends to have the strongest influence on spending.
- **Time on App** also plays an important role.
- The model performs reasonably well with relatively low prediction error.

This workflow represents a typical **machine learning pipeline** used in data science projects.

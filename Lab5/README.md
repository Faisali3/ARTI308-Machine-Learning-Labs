# ARTI 308 – Lab 5: Feature Engineering (Classification)

## Project Overview
This project analyzes a food delivery dataset similar to the Talabat platform.  
The objective is to build a machine learning model that predicts the order status:
whether the order will be **Delivered**, **Cancelled**, or still **In Transit**.

---

## Project Files

| File | Description |
|-----|-------------|
| `ARTI308_Lab5_Complete.ipynb` | The complete Jupyter Notebook containing the analysis and model |
| `talabat_enhanced_orders.csv` | The dataset used in the project |

> Make sure both files are placed in the **same folder** before running the notebook.

---

## Dataset Information

- **100,000 delivery orders**
- **23 columns** including order details, restaurant information, driver data, and customer information
- The dataset is **fully clean**: no missing values and no duplicate rows

### Target Variable: `Order_Status`

| Status | Count | Percentage |
|------|------|------------|
| Delivered | 85,197 | 85.2% |
| Cancelled | 9,812 | 9.8% |
| In Transit | 4,991 | 5.0% |

---

## Step-by-Step Process

### 1. Data Loading and Validation
The dataset was loaded and checked to ensure there were **no missing values, duplicates, or inconsistencies**.

### 2. Feature Engineering

| New Feature | How it was Derived | Purpose |
|-------------|--------------------|--------|
| `order_hour` | Extracted hour from `Order_Time` | Capture time-of-day ordering patterns |
| `order_dayofweek` | Extracted weekday | Identify weekly behavioral patterns |
| `is_weekend` | Indicates Saturday or Sunday | Detect weekend behavior |
| `is_peak_hour` | Lunch (12–15) or dinner (19–23) | Identify high-demand periods |
| `price_per_item` | Total price ÷ quantity | Estimate restaurant price category |
| `haversine_rest_to_cust_km` | Distance between restaurant and customer | Delivery distance |
| `driver_to_restaurant_km` | Distance between driver and restaurant | Driver proximity |
| `price_tier` | Categorized price into low / medium / high / very_high | Simplify price ranges |

The **driver_to_restaurant_km** feature ranked **#1 in feature importance**.

---

### 3. Avoiding Data Leakage

Columns removed:
- `Delivery_Time`
- `Delivery_Duration_Minutes`

These values would not be available at the time an order is placed.

---

### 4. Model Training

The model used is **Random Forest Classifier** with:

- **One-Hot Encoding** for categorical features
- Standard preprocessing steps

Random Forest performs well with **tabular datasets** and captures **non-linear relationships**.

---

## Model Results

**Accuracy: 85.19%**

However, accuracy alone can be misleading because the dataset is **imbalanced**.  
Most orders are labeled as **Delivered**, which biases the model.

---

## Student Tasks

| Task | What Was Done | Result |
|-----|---------------|-------|
| Task 1 | Added `driver_to_restaurant_km` | Ranked **#1 in feature importance** |
| Task 2 | Modified peak hours (added breakfast) | No accuracy change |
| Task 3 | Tested `top_k = 10, 20, 30, 50` | Accuracy stayed the same |
| Task 4 | Used `SelectFromModel` | Simpler model with same accuracy |

---

## Conclusion

- **Geographical features** had the strongest impact on predictions.
- **Feature engineering improved model performance.**
- The main issue in the dataset is **class imbalance**.

### Future Improvement

Apply **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset and improve predictions for minority classes.

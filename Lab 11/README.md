# Lab 11: Credit Card Customer Segmentation with K-Means

## Overview
This project demonstrates the application of **K-Means clustering**, an unsupervised machine learning algorithm, to segment credit card customers based on their usage behavior. The goal is to identify distinct customer groups and develop targeted marketing strategies.

## Project Objectives
- Perform exploratory data analysis on credit card customer behavior data
- Apply K-Means clustering algorithm to segment customers into meaningful groups
- Determine the optimal number of clusters using the Elbow Method and Silhouette Score
- Analyze and interpret customer segments to drive business insights
- Visualize clusters using Principal Component Analysis (PCA)

## Dataset
**Source:** [Kaggle - Credit Card Dataset](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata/data)

**File:** `CC_GENERAL.csv`

### Dataset Features (18 features)
- **CUST_ID**: Customer identifier
- **BALANCE**: Amount of balance on credit card
- **BALANCE_FREQUENCY**: How often the customer uses the card (ratio of months with a balance)
- **PURCHASES**: Total purchase amount on credit card
- **ONEOFF_PURCHASES**: Maximum single purchase amount in a month
- **INSTALLMENTS_PURCHASES**: Purchases made in installments
- **CASH_ADVANCE**: Total cash advance amount
- **PURCHASES_FREQUENCY**: Frequency of purchases
- **ONEOFF_PURCHASES_FREQUENCY**: Frequency of one-off purchases
- **PURCHASES_INSTALLMENTS_FREQUENCY**: Frequency of installment purchases
- **CASH_ADVANCE_FREQUENCY**: Frequency of cash advances
- **CASH_ADVANCE_TRX**: Number of cash advance transactions
- **PURCHASES_TRX**: Number of purchase transactions
- **CREDIT_LIMIT**: Credit limit set for the customer
- **PAYMENTS**: Total amount of payments made
- **MINIMUM_PAYMENTS**: Minimum monthly payment amount
- **PRC_FULL_PAYMENT**: Percentage of full payment made
- **TENURE**: Number of months as a customer

## Project Structure

### 1. **Data Preparation**
   - Import necessary libraries (pandas, scikit-learn, matplotlib, seaborn)
   - Load and explore the dataset
   - Drop non-behavioral identifiers (CUST_ID)
   - Handle missing values using mean imputation

### 2. **Exploratory Data Analysis (EDA)**
   - Distribution analysis with histograms
   - Correlation heatmap to understand feature relationships
   - Scatter plots between key features (Balance vs Purchases, Balance vs Cash Advance)

### 3. **Feature Scaling**
   - Apply StandardScaler to normalize features
   - Essential for distance-based K-Means algorithm

### 4. **Determining Optimal K**
   - **Elbow Method**: Plot inertia values for K=1 to 10
   - **Silhouette Score**: Evaluate cluster separation quality for K=2 to 10
   - Choose K=3 as the optimal number of clusters

### 5. **K-Means Clustering**
   - Train final K-Means model with K=3
   - Assign cluster labels to customers
   - Analyze cluster centers and characteristics

### 6. **Cluster Analysis & Interpretation**
   - Generate summary statistics for each cluster
   - Identify customer segment characteristics
   - Provide business insights and recommendations

### 7. **Visualization**
   - PCA (Principal Component Analysis) reduces 18 features to 2 for visualization
   - 2D scatter plot shows cluster separation and centroid locations

## Key Findings

### Identified Customer Segments:

**Cluster 0 - Active Purchasers**
- High purchase amounts and frequency
- Regular card usage
- Consistent payment patterns
- **Classification:** High-value customers

**Cluster 1 - Cash Advance Users**
- High cash advance amounts
- Frequent cash advance usage
- Lower purchase activity
- **Classification:** Risk-oriented segment

**Cluster 2 - Inactive/Low Activity**
- Low transaction volume
- Minimal card usage
- Low balances and purchases
- **Classification:** Disengaged customers

## Business Applications

### Marketing Strategy
1. **Cluster 0**: Rewards programs, cashback incentives, premium card tiers
2. **Cluster 1**: Competitive cash advance rates, personal loan products
3. **Cluster 2**: Engagement campaigns, promotional offers, financial education

### Risk Management
- Different credit policies for each segment
- Tailored payment plans and credit limits
- Fraud detection strategies specific to behavior patterns

### Product Development
- Design products aligned with segment needs
- Cross-selling opportunities across segments
- Customer lifecycle management programs

## Methodology

### K-Means Algorithm
- **Algorithm Type**: Unsupervised clustering
- **Distance Metric**: Euclidean distance
- **Initialization**: K-means++ (via n_init=10)
- **Random State**: 42 (for reproducibility)

### Evaluation Metrics
1. **Inertia**: Sum of squared distances from points to their cluster centers
2. **Silhouette Score**: Measures how similar objects are to their cluster (-1 to 1)
3. **Cluster Size**: Distribution of customers across segments

## Why Unsupervised Learning?
- **No predefined labels**: Customer groups are unknown beforehand
- **Pattern discovery**: Algorithm finds natural groupings in data
- **Exploratory analysis**: Reveals hidden customer behavior patterns
- **Actionable insights**: Segments inform business strategy

## Technologies Used
- **Python 3.11.4**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms (KMeans, StandardScaler, PCA, silhouette_score)
- **matplotlib & seaborn**: Data visualization
- **Jupyter Notebook**: Interactive analysis and documentation

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Notebook
```bash
jupyter notebook Lab11-Credit_Card_Customer_Segmentation.ipynb
```

### Steps
1. Ensure `CC_GENERAL.csv` is in the same directory
2. Run all cells sequentially
3. Analyze outputs and visualizations
4. Answer the final questions based on your findings

## Key Insights & Recommendations

### Data Preprocessing
- Mean imputation handled missing values while preserving data distribution
- StandardScaler normalization essential for distance-based clustering

### Optimal K Selection
- K=3 balances simplicity, statistical metrics, and business interpretability
- Elbow point occurs at K=3, indicating diminishing returns beyond this point
- Silhouette score of ~0.35 indicates reasonable cluster separation

### Cluster Characteristics
- **Size Distribution**: Clusters are reasonably balanced (no extreme imbalance)
- **Separation**: PCA visualization shows distinct cluster regions
- **Interpretability**: Each cluster represents meaningful customer behavior pattern

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Feature scaling | StandardScaler normalization |
| Determining K | Elbow method + Silhouette score |
| High dimensionality | PCA for 2D visualization |
| Missing values | Mean imputation |
| Cluster interpretation | Detailed feature analysis and business context |

## Future Improvements

1. **Advanced Clustering**: Test DBSCAN, Hierarchical clustering, or Gaussian Mixture Models
2. **Feature Engineering**: Create interaction features or polynomial features
3. **Temporal Analysis**: Incorporate time-series customer behavior patterns
4. **Deep Learning**: Use autoencoders for non-linear dimensionality reduction
5. **Dynamic Segmentation**: Update clusters periodically as new customer data arrives

## References
- Scikit-learn Documentation: https://scikit-learn.org/
- K-Means Clustering: https://en.wikipedia.org/wiki/K-means_clustering
- Silhouette Score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- PCA: https://scikit-learn.org/stable/modules/decomposition.html#pca

## Author
ARTI308 - Machine Learning Course

## License
This project is provided for educational purposes.

---

**Last Updated:** 2024
**Status:** Complete

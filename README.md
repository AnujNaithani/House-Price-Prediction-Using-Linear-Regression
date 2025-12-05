# Linear Regression Model for predicting the house price prediction

This project implements a **Linear Regression model** to predict a target variable based on multiple features.  
The workflow includes data exploration, preprocessing, train–test splitting, model training, evaluation, and visualization.


## Project Structure
- `Linear Regression Model.ipynb` — Main notebook containing the complete workflow  
- `Housing.csv`  — Dataset used for training/testing  
- `README.md` — Project documentation  


## Dataset Overview
- Loaded dataset using pandas 
- Dataset have 545 rows and 13 columns 
- No missing values found  
- No duplicate rows found  
- One column (`hotwaterheating`) was **highly imbalanced** (520 “no” out of 545 samples)


## Steps Performed

### **1. Importing Libraries**
Used essential libraries:
- pandas  
- NumPy  
- matplotlib  
- seaborn  
- scikit-learn  


### **2. Data Exploration**
- Checked data info and summary statistics  
- Verified missing values  
- Verified duplicated rows  
- Checked distribution of categorical columns  
- Identified imbalance in the `hotwaterheating` feature and having a large difference in median values for both classes


### **3. Train–Test Split**
Before Train-Test Split map binary categorical columns<br>

Used **StratifiedShuffleSplit** to maintain the distribution of `hotwaterheating`:

```python
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
```

### **4. Handle Outlier**
- Handle outlier in area column using IQR method and performing capping

### **5. Ordinal Encoding**
- perform Ordinal Encoding on furnishingstatus column with order **['unfurnished','semi-furnished','furnished']**

### **6. Trained Linear Regression Model**

### **7. Evaluate Model**
- Evaluate model using R2 score, MAE and RMSE
- R2: 0.7026299010925842
- MAE: 730846.9802995496
- RMSE: 993618.5078248533

### **8. Cross validation**
I evaluated the model using **cross-validation** to measure how consistently the model performs across different subsets of the data.

I used **K-Fold cross-validation** with RMSE as the evaluation metric:

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,x_train, y_train,scoring='neg_root_mean_squared_error',cv=10)
rmse_score = -scores
```
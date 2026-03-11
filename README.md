# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the HeightsWeights.csv dataset using pandas.
2. Select the features Height (Inches) and Weight (Pounds) from the dataset.
3. Visualize the original data distribution using a scatter plot.
4. Standardize the data using StandardScaler and apply PCA (Principal Component Analysis) with 2 components.
5. Display the explained variance ratio and visualize the transformed data using a PCA scatter plot.


## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: Suwasthika V
RegisterNumber: 212225040445
*/
#import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
#load the dataset
data=pd.read_csv('HeightsWeights.csv')
print("First 5 rows of the dataset:")
print(data.head())
#select features
X=data[['Height(Inches)','Weight(Pounds)']]
#visualize original data
plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)',y='Weight(Pounds)',data=data)
plt.title("Original Data Distribution")
plt.show()
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)
print("Explained Variance Ratio:",pca.explained_variance_ratio_)
pca_df=pd.DataFrame(X_pca,columns=['PC1','PC2'])
plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1',y='PC2',data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

## Output:
<img width="767" height="177" alt="image" src="https://github.com/user-attachments/assets/85579e80-218e-457c-b723-d9e5a19a6b0f" />
<img width="896" height="605" alt="image" src="https://github.com/user-attachments/assets/8b8fe5a7-3148-4ba3-bf79-8424993fd2ad" />
<img width="670" height="49" alt="image" src="https://github.com/user-attachments/assets/9948895f-2954-4bc8-8bdb-20c0949e355b" />
<img width="838" height="593" alt="image" src="https://github.com/user-attachments/assets/f70b85a5-2583-4158-a69e-d2da43244ec8" />

## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.

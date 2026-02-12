# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Read the Dataset
# Replace with your actual CSV file
df = pd.read_csv('Data_set.csv')
df.head()

# Step 3: Dataset Information
df.info()
df.describe()

# Step 4: Handling Missing Values
# Check Null Values
df.isnull()
df.isnull().sum()

# Fill Missing Values with 0
df1_fill_0 = df.fillna(0)
df1_fill_0

# Forward Fill
df1_ffill = df.ffill()
df1_ffill

# Backward Fill
df1_bfill = df.bfill()
df1_bfill

# Fill with Mean (Numerical Column Example)
df['rating']=df['rating'].fillna(df['rating'].mean())
df['watchers']=df['watchers'].fillna(df['watchers'].mean())
df

# Drop Missing Values
df1_dropna = df.dropna()
df1_dropna

# Step 5: Save Cleaned Data
df1_dropna.to_csv('Data_set1.csv', index=False)

# OUTLIER DETECTION
# Step 6: IQR Method (Using Dataset)
df1 = pd.read_csv('Data_set.csv')

# Boxplot for Outlier Detection
sns.boxplot(x=df1['watchers'])
plt.show()

# Calculate IQR
Q1 = df['watchers'].quantile(0.25)
Q3 = df['watchers'].quantile(0.75)
IQR = Q3-Q1
print("IQR:", IQR)

# Detect Outliers
outliers = df[
(df['watchers'] < (Q1-1.5 * IQR)) |
(df['watchers'] > (Q3 + 1.5 * IQR))
]
outliers

# Remove Outliers
df1_cleaned = df[
~((df['watchers'] < (Q1-1.5 * IQR)) |
(df['watchers'] > (Q3 + 1.5 * IQR)))
]
df1_cleaned

data = [1,12,15,18,21,24,27,30,33,36,39,42,45,48,51,
54,57,60,63,66,69,72,75,78,81,84,87,90,93]
df_z = pd.DataFrame(data, columns=['values'])
df_z

# Calculate Z-Scores
z_scores = np.abs(stats.zscore(df_z))
z_scores

threshold=3
mask = np.zeros(len(df), dtype=bool)
mask[df['rating'].dropna().index] = z_score > threshold
outliers = df[mask]
print('outliers')
print(outliers)

df_z_cleaned = df_z[z_scores <= threshold]
df_z_cleaned

```







# Result

<img width="668" height="671" alt="Screenshot 2026-02-12 082239" src="https://github.com/user-attachments/assets/1880767e-b8f5-4495-bc1d-56e8e02b98e8" />


<img width="468" height="226" alt="Screenshot 2026-02-12 082259" src="https://github.com/user-attachments/assets/7f9b151b-c41f-4e1c-a7b1-4f475c8a043f" />


<img width="456" height="813" alt="Screenshot 2026-02-12 082323" src="https://github.com/user-attachments/assets/11117d15-bc57-476d-82a8-f76ea3a370e8" />









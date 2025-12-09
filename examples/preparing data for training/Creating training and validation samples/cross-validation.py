import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv("salary_data.csv")

X = df['YearsExperience'].values.reshape(-1, 1)
y = df['Salary'].values

# Define the number of folds for cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

lin_reg = LinearRegression()

scores = cross_val_score(lin_reg, X, y, cv=kf, scoring='r2')
print(scores)

mean_score = np.mean(scores)
print(mean_score)
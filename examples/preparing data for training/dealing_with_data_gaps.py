import opendatasets as od
import pandas as pd
import numpy as np

od.download("https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=train.csv")

# Теперь исследуем датасет на наличие пропусков. Это можно сделать несколькими способами.
# Способ первый:
print("\nСпособ первый:")
df = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv')
df.info()

# Способ второй:
print("\nСпособ второй:")
print(df.isnull().sum())

# Эта конструкция выбирает все строки (:) и только те столбцы, в которых есть хотя бы одно пропущенное значение (NaN).
df_nan = df.loc[:, df.isnull().any()]
# посмотрим процент пропущенных значений
print("\nПроцент пропущенных значений:")
print(df_nan.isnull().sum() / len(df_nan) * 100)

# Видим, что есть столбцы, в которых более 80% пропусков, — такие переменные лучше в целом удалить,
# так как заполнение может привести в дальнейшем только к ухудшению результата.
# Удалим данные переменные (столбцы)
df = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis = 1)

# Заполним одну из переменных константным значением равным 0
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
# df["MasVnrArea"] = df["MasVnrArea"].replace(np.nan, 0)

# Заполним другую переменную, используя среднее значение по всей переменной или медиану
#using mean
df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)
# using median
# df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)

# Заполним с использованием моды для нечисловых переменных
# using mode
df["GarageFinish"].fillna(df["GarageFinish"].mode()[0], inplace=True)

# Заполненим, используя предыдущие или последующие значения в строке
# using ffill method
df["GarageType"] = df["GarageType"].fillna(method='ffill')
# OR
# using bfill method
df["Electrical"] = df["Electrical"].fillna(method='bfill')

# Заполним с использованием методов интерполяции
df["GarageYrBlt"].interpolate(method="linear", direction = "forward", inplace=True)

print("После обработки данных:")
df.info()

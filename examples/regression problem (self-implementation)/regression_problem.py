import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


class LinearRegression():

    def __init__(self, learning_rate, iterations):
        self.theta = None
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        # классический метод для тренировки
        theta = np.zeros(2)  # [theta0, theta1]
        X_b = np.c_[np.ones(len(X)), X]  # добавляем столбец единиц для theta0

        for i in range(self.iterations):
            # здесь код изменения весов, правильнее его вынести в отдельный метод
            theta = self.update_weights(X_b, theta, Y)

        self.theta = theta  # ← сохраняем в self.theta
        return self  # ← по соглашению, fit() возвращает сам объект

    def update_weights(self, X_b, theta, Y):
        # здесь код для изменения весов
        m = X_b.shape[0]  # количество примеров
        grad = (2 / m) * X_b.T.dot(X_b.dot(theta) - Y)
        theta = theta - self.learning_rate * grad

        return theta

    def predict(self, X):
        # а тут предсказание значений
        X = np.atleast_1d(X)  # гарантирует массив
        X_new = np.c_[np.ones(len(X)), X]

        return X_new.dot(self.theta)


df = pd.read_csv("salary_data.csv")

X = df['YearsExperience'].values
Y = df['Salary'].values

model = LinearRegression(iterations=1000, learning_rate=0.01)
model.fit(X, Y)

Y_pred = model.predict(X)

plt.scatter(X, Y, color='blue', label='Исходные данные')
plt.plot(X, Y_pred, color='orange', label='Линия регрессии')
plt.title('Зависимость зарплаты от опыта')
plt.xlabel('Число лет опыта')
plt.ylabel('Зарплата')
plt.legend()

plt.show()

print('Mean Absolute Error:', mean_absolute_error(Y, Y_pred))
print('Mean Squared Error:', mean_squared_error(Y, Y_pred))
print('R2 score:', r2_score(Y, Y_pred))

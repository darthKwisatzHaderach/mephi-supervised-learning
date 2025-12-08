import numpy as np
from matplotlib import pyplot as plt

# Данные: рост (в метрах), вес (в кг)
X = np.array([1.5, 1.6, 1.7, 1.8, 1.9])
y = np.array([50, 58, 68, 76, 85])

m = len(y)
theta = np.zeros(2)  # [theta0, theta1]
X_b = np.c_[np.ones(m), X]  # добавляем столбец единиц для theta0
eta = 0.1
for _ in range(1000):
    grad = (2/m) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * grad

print("theta0 =", theta[0], ", theta1 =", theta[1])
theta0, theta1 = theta
print(f"Обученная модель: вес = {theta0:.2f} + {theta1:.2f} × рост")


# 4. Функция для предсказания на новых данных
def predict_weight(new_heights, theta):
    """
    Применяет обученную модель к новым значениям роста.

    Параметры:
        new_heights: float или array-like — рост(ы) в метрах
        theta: array-like — [theta0, theta1], обученные параметры

    Возвращает:
        предсказанный вес (или массив весов)
    """
    new_heights = np.atleast_1d(new_heights)  # гарантирует массив
    X_new = np.c_[np.ones(len(new_heights)), new_heights]
    return X_new.dot(theta)


# 5. Пример применения к новым данным
new_heights = [1.65, 1.85, 2.0]  # новые росты
predicted_weights = predict_weight(new_heights, theta)

print("\nПредсказания для новых данных:")
for h, w in zip(new_heights, predicted_weights):
    print(f"Рост {h:.2f} м → предсказанный вес: {w:.2f} кг")

# 6. (Опционально) график
x_plot = np.linspace(1.4, 2.0, 100)
y_plot = predict_weight(x_plot, theta)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', label='Обучающие данные', s=60)
plt.plot(x_plot, y_plot, color='blue', label='Модель')
plt.scatter(new_heights, predicted_weights, color='green', label='Новые предсказания', s=80, zorder=5)
plt.xlabel('Рост (м)')
plt.ylabel('Вес (кг)')
plt.title('Линейная регрессия + предсказание на новых данных')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

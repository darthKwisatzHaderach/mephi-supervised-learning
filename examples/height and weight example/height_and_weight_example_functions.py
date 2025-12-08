import matplotlib.pyplot as plt

# 1. Исходные данные
X = [1.5, 1.6, 1.7, 1.8, 1.9]
y = [50, 58, 68, 76, 85]
m = len(X)

# 2. Начальные значения параметров
theta0 = 0.0
theta1 = 0.0

# 3. Гиперпараметры
learning_rate = 0.1
num_epochs = 1000

# 4. Функция предсказания
def predict(x, theta0, theta1):
    return theta0 + theta1 * x

# 5. Функция потерь (MSE)
def compute_loss(X, y, theta0, theta1):
    total_error = 0.0
    for i in range(len(X)):
        pred = predict(X[i], theta0, theta1)
        error = pred - y[i]
        total_error += error ** 2
    return total_error / (2 * len(X))

# 6. Обучение с градиентным спуском
for epoch in range(num_epochs):
    # Накопители градиентов
    d_theta0 = 0.0
    d_theta1 = 0.0

    # Считаем градиенты вручную
    for i in range(m):
        pred = predict(X[i], theta0, theta1)
        error = pred - y[i]
        d_theta0 += error
        d_theta1 += error * X[i]

    # Усредняем градиенты
    d_theta0 /= m
    d_theta1 /= m

    # Обновляем параметры
    theta0 = theta0 - learning_rate * d_theta0
    theta1 = theta1 - learning_rate * d_theta1

    # (опционально) выводим прогресс каждые 200 эпох
    if epoch % 200 == 0:
        loss = compute_loss(X, y, theta0, theta1)
        print(f"Эпоха {epoch}: loss = {loss:.4f}, theta0 = {theta0:.3f}, theta1 = {theta1:.3f}")

# 7. Итог
print("\nОкончательная модель:")
print(f"вес ≈ {theta0:.2f} + {theta1:.2f} × рост")

# 8. Построение графика
# Генерируем плавную линию для модели
# Можно без numpy — просто через цикл
x_line = []
y_line = []
x_start, x_end = min(X), max(X)
steps = 100
dx = (x_end - x_start) / steps
for i in range(steps + 1):
    x_val = x_start + i * dx
    x_line.append(x_val)
    y_line.append(predict(x_val, theta0, theta1))

# Строим
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', label='Исходные данные', s=50)
plt.plot(x_line, y_line, color='blue', label=f'Модель: y = {theta0:.2f} + {theta1:.2f}·x')
plt.xlabel('Рост (м)')
plt.ylabel('Вес (кг)')
plt.title('Линейная регрессия: подбор зависимости веса от роста')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
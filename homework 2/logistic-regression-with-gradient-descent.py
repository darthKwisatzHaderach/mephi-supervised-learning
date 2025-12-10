import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def log_loss(w, X, y):
    # здесь должны быть рассчитаны параметры градиента и значение функции потерь
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X))

    loss = -1.0 / m * np.sum(y * np.log(A) + (1 - A) * np.log(1 - A))
    loss = np.squeeze(loss)

    grad = 1.0 / m * np.dot(X, (A - y).T)

    return loss, grad


def optimize(w, X, y, n_iterations, eta):
    # потери будем записывать в список для отображения в виде графика
    losses = []

    for i in range(n_iterations):
        # считаем веса
        loss, grad = log_loss(w, X, y)

        w = w - eta * grad

        losses.append(loss)

    return w, losses


def predict(w, X, b=0.5):
    m = X.shape[1]

    y_predicted = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X))

    # За порог отнесения к тому или иному классу примем вероятность 0.5
    for i in range(A.shape[1]):
        if A[:, i] > b:
            y_predicted[:, i] = 1
        elif A[:, i] <= b:
            y_predicted[:, i] = 0
    return y_predicted

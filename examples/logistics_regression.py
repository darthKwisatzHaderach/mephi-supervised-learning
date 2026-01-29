from sklearn import datasets
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

# сгенерируем данные
classes = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=1)

# Параметры datasets.make_classification():
# - n_samples=100 — количество образцов (100 точек данных)
# - n_features=2 — количество признаков (2 признака, двумерные данные)
# - n_informative=2 — количество информативных признаков (оба признака полезны для классификации)
# - n_redundant=0 — количество избыточных признаков (0, нет линейных комбинаций информативных)
# - n_classes=2 — количество классов (2 класса, бинарная классификация)
# - random_state=1 — зерно генератора случайных чисел (для воспроизводимости результатов)
# Что получается:
# - 100 точек в 2D пространстве
# - 2 признака, оба информативны
# - 2 класса (красный и синий на графике)
# Данные воспроизводимы благодаря random_state

# и изобразим их на графике
colors = ListedColormap(['red', 'blue'])

plt.figure(figsize=(8, 8))
plt.scatter([x[0] for x in classes[0]], [x[1] for x in classes[0]], c=classes[1], cmap=colors)
plt.show()

# перемешивание датасета
np.random.seed(12) # это число позволяет постоянно получать одну и ту же "случайность"
shuffle_index = np.random.permutation(classes[0].shape[0])
X_shuffled, y_shuffled = classes[0][shuffle_index], classes[1][shuffle_index]

# разбивка на обучающую и тестовую выборки
train_proportion = 0.7
train_test_cut = int(len(classes[0]) * train_proportion)

X_train, X_test, y_train, y_test = \
    X_shuffled[:train_test_cut], \
    X_shuffled[train_test_cut:], \
    y_shuffled[:train_test_cut], \
    y_shuffled[train_test_cut:]

print("Размер массива признаков обучающей выборки", X_train.shape)
print("Размер массива признаков тестовой выборки", X_test.shape)
print("Размер массива ответов для обучающей выборки", y_train.shape)
print("Размер массива ответов для тестовой выборки", y_test.shape)

X_train_tr = X_train.transpose()
y_train_tr = y_train.reshape(1, y_train.shape[0])
X_test_tr = X_test.transpose()
y_test_tr = y_test.reshape(1, y_test.shape[0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_loss(w, X, y):
    # здесь должны быть рассчитаны параметры градиента и значение функции потерь

    return loss, grad

def optimize(w, X, y, n_iterations, eta):
    # потери будем записывать в список для отображения в виде графика
    losses = []

    for i in range(n_iterations):
        # считаем веса

        losses.append(loss)

    return w, losses

def predict(w, X, b=0.5):

    # За порог отнесения к тому или иному классу примем вероятность 0.5
    for i in range(A.shape[1]):

    return y_predicted
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * x - 6 * x + 5

def dfdx(x):
    return 2 * x - 6

def f2(x):
  return x*x + 15*np.sin(x)

def df2dx(x):
  return 2*x + 15*np.cos(x)

def gd(N, lr, min=-5, max=11, f=f, dfdx=dfdx, xc=-4):
    """
    Реализация и визуализация процедуры градиентного спуска
    для конкретной функции в упрощенном примере.

    Parameters
    ----------
    N: integer
        Число шагов градиентного спуска, которые необходимо выполнить.

    lr: float
        Скорость обучения

    min: float
        Минимальное значение, отображаемое на графике

    max: float
        Максимальное значение, отображаемое на графике

    f: function
        Целевая функция

    dfdx: function
        Градиент целевой функции

    Returns
    ----------
    """
    fig, ax = plt.subplots()
    ax.set_title(f'Скорость движения: {lr}, количество итераций: {N}')

    x_plot = np.arange(min, max, 0.01)
    y_plot = [f(x) for x in x_plot]
    ax.plot(x_plot, y_plot)

    print(f'Начальное значение минимума: {f(xc)} в точке {xc}')

    ax.scatter(xc, f(xc), c='r')

    for _ in range(N):
        x0 = xc  # нужно только для графика

        xc = xc - lr * dfdx(xc)  # сам по себе градиентный спуск

        ax.scatter(xc, f(xc), c='b')  # нужно только для графика
        ax.plot([x0, xc], [f(x0), f(xc)], c='r')  # нужно только для графика

    ax.scatter(xc, f(xc), c='g')

    print(f'Финальное значение минимума: {f(xc)} в точке {xc}')
    plt.show()

#gd(200, 1.1, min=-5, max=11, f=f)

#gd(20, 1, min=-5, max=11, f=f)

#for i in range(1, 10):
#    gd(N=20, lr=i / 10, f=f)

gd(20, 0.05, min=-15, max= 15, f=f2, dfdx=df2dx, xc=10)
gd(20, 0.03, min=-15, max=15, f=f2, dfdx=df2dx, xc=-5)
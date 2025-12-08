import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 20)
y = 2 * x + 3 + np.random.normal(0, 2, size=x.shape)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(x, y)
plt.title('Левый график')

plt.subplot(1, 2, 2)
plt.scatter(x, y)
plt.title('Правый график')

plt.suptitle('Функциональный стиль')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(x, y)
axes[0].set_title('Левый график')

axes[1].scatter(x, y)
axes[1].set_title('Правый график')

fig.suptitle('ООП стиль')
fig.tight_layout()
plt.show()
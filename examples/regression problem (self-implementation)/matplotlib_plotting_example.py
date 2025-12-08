import matplotlib.pyplot as plt
import numpy as np

# Минимальные данные
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 1, 5, 3]
y2 = [1, 3, 2, 4, 5]
y3 = [np.sin(i) for i in np.linspace(0, 2*np.pi, 5)]

# Создаём фигуру и три оси: 1 строка, 3 столбца
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# ─── График 1: точечная диаграмма ───────────────────────
axes[0].scatter(x, y1, color='blue', s=50)
axes[0].set_title('Точки')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y1')
axes[0].grid(True, linestyle='--', alpha=0.5)

# ─── График 2: линейный график ──────────────────────────
axes[1].plot(x, y2, color='green', marker='o', linewidth=2)
axes[1].set_title('Линия с маркерами')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y2')
axes[1].grid(True, linestyle='--', alpha=0.5)

# ─── График 3: гистограмма ──────────────────────────────
axes[2].bar(x, y3, color='purple', alpha=0.7)
axes[2].set_title('Столбцы')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y3')
axes[2].grid(True, axis='y', linestyle='--', alpha=0.5)

# ─── Общие настройки ────────────────────────────────────
fig.suptitle('ООП-стиль: три разных графика', fontsize=14)
fig.tight_layout()  # автоматически подгоняет отступы

# Показываем
plt.show()
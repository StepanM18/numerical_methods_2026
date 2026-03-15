import csv
import numpy as np
import matplotlib.pyplot as plt

# 1. ЗЧИТУВАННЯ ДАНИХ З CSV
def read_data(filename):
    x = []
    y = []

    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Objects']))
            y.append(float(row['FPS']))

    return np.array(x), np.array(y)


x, y = read_data("data.csv")

print("1) Зчитані дані:")
print("Objects =", x)
print("FPS =", y)


# 2. ПОБУДОВА ТАБЛИЦІ РОЗДІЛЕНИХ РІЗНИЦЬ
def divided_difference_table(x, y):
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table


table = divided_difference_table(x, y)

print("\nТаблиця розділених різниць:")
print("=" * 90)
print(f"{'x':>8} |   {'f(x)':>10} | {'1 порядок':>12} | {'2 порядок':>12} | {'3 порядок':>12} | {'4 порядок':>12}")
print("=" * 90)

for i in range(len(x)):
    print(f"{x[i]:8.0f} | ", end="")
    for j in range(len(x) - i):
        print(f"{table[i][j]:12.4e} | ", end="")
    print()

# 3. ФУНКЦІЯ ІНТЕРПОЛЯЦІЇ НЬЮТОНА
def newton_polynomial(x, table, value):
    n = len(x)
    result = table[0][0]

    product = 1
    for i in range(1, n):
        product *= (value - x[i - 1])
        result += table[0][i] * product

    return result


# 4. ПРОГНОЗ ДЛЯ 1000 ОБ'ЄКТІВ
fps_1000 = newton_polynomial(x, table, 1000)

print("\n3) Прогноз FPS для 1000 об'єктів:")
print("FPS(1000) =", round(fps_1000, 2))


# 5. ВИЗНАЧЕННЯ ОПТИМАЛЬНОГО НАВАНТАЖЕННЯ
target_fps = 60

# Будуємо таблицю розділених різниць для зворотної залежності (FPS -> об'єкти)
table_inv = divided_difference_table(y, x)

# Обчислюємо методом Ньютона
n = len(y)
result = table_inv[0][0]  # починаємо з x(0)
product = 1
for i in range(1, n):
    product *= (target_fps - y[i - 1])
    result += table_inv[0][i] * product

objects_for_60fps = result

print("\n" + "="*80)
print(f"МІНІМАЛЬНА КІЛЬКІСТЬ ОБ'ЄКТІВ ДЛЯ FPS ≥ {target_fps}")
print("="*80)
print(f" {objects_for_60fps:.0f} об'єктів дадуть FPS ≈ {target_fps}")

# 6. ПОБУДОВА ГРАФІКА (ОСНОВНИЙ)
x_plot = np.linspace(min(x), max(x), 500)
y_plot = [newton_polynomial(x, table, val) for val in x_plot]

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', s=100, label='Дані')
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Інтерполяція Ньютона')
plt.scatter([1000], [fps_1000], color='green', s=200, marker='*', label=f'Прогноз: {fps_1000:.1f} FPS')
plt.title("Інтерполяція методом Ньютона")
plt.xlabel("Кількість об'єктів")
plt.ylabel("FPS")
plt.legend()
plt.grid()
plt.savefig('fps_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. ДОСЛІДЖЕННЯ ВПЛИВУ КІЛЬКОСТІ ВУЗЛІВ ТА ГРАФІК ПОХИБОК
print("\n5) Дослідження впливу кількості вузлів та похибок:")

# Створюємо фігуру з двома графіками поряд
plt.figure(figsize=(14, 6))

# ===== ГРАФІК 1: Інтерполяція =====
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='red', s=100, label='Дані')

for nodes in [3, 4, 5]:
    x_subset = x[:nodes]
    y_subset = y[:nodes]
    temp_table = divided_difference_table(x_subset, y_subset)
    prediction = newton_polynomial(x_subset, temp_table, 1000)
    print(f"Вузлів = {nodes}, FPS(1000) = {round(prediction, 2)}")

    x_plot_subset = np.linspace(min(x), max(x), 500)
    y_plot_subset = [newton_polynomial(x_subset, temp_table, val) for val in x_plot_subset]
    plt.plot(x_plot_subset, y_plot_subset, '--', label=f'{nodes} вузлів')

plt.title("Вплив кількості вузлів на інтерполяцію")
plt.xlabel("Кількість об'єктів")
plt.ylabel("FPS")
plt.legend()
plt.grid()

# ===== ГРАФІК 2: Похибки =====
plt.subplot(1, 2, 2)

# Еталон - інтерполяція з 5 вузлами
x_plot = np.linspace(min(x), max(x), 500)
y_true = [newton_polynomial(x, table, val) for val in x_plot]

for nodes in [3, 4]:
    x_subset = x[:nodes]
    y_subset = y[:nodes]
    temp_table = divided_difference_table(x_subset, y_subset)
    y_plot = [newton_polynomial(x_subset, temp_table, val) for val in x_plot]

    # Похибка = |інтерполяція з nodes вузлів - інтерполяція з 5 вузлів|
    error = np.abs(np.array(y_plot) - np.array(y_true))
    plt.plot(x_plot, error, label=f'Похибка для {nodes} вузлів')

# Додаємо нульову лінію для 5 вузлів (опціонально)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5, label='5 вузлів (еталон)')

plt.title("Графік похибок інтерполяції")
plt.xlabel("Кількість об'єктів")
plt.ylabel("Похибка (FPS)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('nodes_influence.png', dpi=300, bbox_inches='tight')
plt.show()

# Виводимо максимальні похибки
print("\nМаксимальні похибки відносно 5 вузлів:")
for nodes in [3, 4]:
    x_subset = x[:nodes]
    y_subset = y[:nodes]
    temp_table = divided_difference_table(x_subset, y_subset)
    y_plot = [newton_polynomial(x_subset, temp_table, val) for val in x_plot]
    error = np.max(np.abs(np.array(y_plot) - np.array(y_true)))
    print(f"{nodes} вузли: максимальна похибка = {error:.2f} FPS")
# 8. ЕФЕКТ РУНГЕ
print("\n5) Демонстрація ефекту Рунге(графік)")

# Створюємо НОВИЙ графік для ефекту Рунге
plt.figure(figsize=(10, 6))

x_runge = np.linspace(-5, 5, 10)
y_runge = 1 / (1 + x_runge ** 2)

table_runge = divided_difference_table(x_runge, y_runge)

x_dense = np.linspace(-5, 5, 500)
y_dense = 1 / (1 + x_dense ** 2)
y_interp = [newton_polynomial(x_runge, table_runge, val) for val in x_dense]

plt.plot(x_dense, y_dense, 'k-', linewidth=2, label='Точна функція')
plt.plot(x_dense, y_interp, 'r--', linewidth=2, label='Інтерполяція (10 вузлів)')
plt.scatter(x_runge, y_runge, color='blue', s=50, label='Вузли')
plt.title("Ефект Рунге")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.savefig('runge_effect.png', dpi=300, bbox_inches='tight')
plt.show()


# 9. ПОРІВНЯННЯ З ЛАГРАНЖЕМ

def lagrange(x, y, value):
    total = 0
    n = len(x)

    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (value - x[j]) / (x[i] - x[j])
        total += term

    return total


lagrange_result = lagrange(x, y, 1000)

print("\n6) Порівняння методів:")
print("Ньютон =", round(fps_1000, 2))
print("Лагранж =", round(lagrange_result, 2))

# Створюємо НОВИЙ графік для порівняння
plt.figure(figsize=(10, 6))

x_plot = np.linspace(min(x), max(x), 500)  # створюємо x_plot заново
y_plot_newton = [newton_polynomial(x, table, val) for val in x_plot]  # перераховуємо Ньютон
y_lagrange_plot = [lagrange(x, y, val) for val in x_plot]

plt.scatter(x, y, color='red', s=100, label='Дані')
plt.plot(x_plot, y_plot_newton, 'b-', linewidth=2, label='Ньютон')  # тепер правильно!
plt.plot(x_plot, y_lagrange_plot, 'g--', linewidth=2, label='Лагранж')
plt.title("Порівняння методів Ньютона та Лагранжа")
plt.xlabel("Кількість об'єктів")
plt.ylabel("FPS")
plt.legend()
plt.grid()
plt.savefig('methods_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# 10. ДОСЛІДЖЕННЯ ВПЛИВУ КРОКУ
print("\n" + "=" * 80)
print("ДОСЛІДЖЕННЯ ВПЛИВУ КРОКУ(графік)")
print("=" * 80)


# Тестова функція для демонстрації
def test_func(x):
    return 120 * np.exp(-x / 800) + 20


a, b = 0, 1600  # фіксований інтервал
nodes_counts = [5, 9, 17]

plt.figure(figsize=(15, 5))

for idx, n_nodes in enumerate(nodes_counts):
    x_test = np.linspace(a, b, n_nodes)
    y_test = test_func(x_test)
    h = (b - a) / (n_nodes - 1)

    table_test = divided_difference_table(x_test, y_test)
    x_plot = np.linspace(a, b, 500)
    y_plot = [newton_polynomial(x_test, table_test, val) for val in x_plot]
    y_true = test_func(x_plot)

    plt.subplot(1, 3, idx + 1)
    plt.plot(x_plot, y_true, 'k-', label='Точна')
    plt.plot(x_plot, y_plot, 'r--', label='Інтерполяція')
    plt.scatter(x_test, y_test, color='blue', s=30)
    plt.title(f'{n_nodes} вузлів, h={h:.1f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

plt.suptitle('Дослідження впливу кроку (фіксований інтервал [0,1600])')
plt.tight_layout()
plt.savefig('step_influence.png', dpi=300, bbox_inches='tight')
plt.show()

#11. ДОСЛІДЖЕННЯ ВПЛИВУ КІЛЬКОСТІ ВУЗЛІВ (фіксований крок)
print("\n" + "=" * 80)
print("2) Дослідження впливу кількості вузлів (фіксований крок h=200)")
print("=" * 80)


# Тестова функція (та сама, що й у першому дослідженні)
def test_func(x):
    return 120 * np.exp(-x / 800) + 20


h_fixed = 200  # фіксований крок
intervals = [800, 1600, 2400]  # різні інтервали

plt.figure(figsize=(15, 5))

for idx, interval in enumerate(intervals):
    # Кількість вузлів = довжина інтервалу / крок + 1
    n_nodes = interval // h_fixed + 1

    # Створюємо вузли
    x_test = np.linspace(0, interval, n_nodes)
    y_test = test_func(x_test)

    # Будуємо інтерполяцію
    table_test = divided_difference_table(x_test, y_test)
    x_plot = np.linspace(0, interval, 500)
    y_plot = [newton_polynomial(x_test, table_test, val) for val in x_plot]
    y_true = test_func(x_plot)

    # Обчислюємо похибку
    error = np.max(np.abs(np.array(y_plot) - y_true))
    print(f"Інтервал [0,{interval}], вузлів: {n_nodes}, макс. похибка: {error:.4f}")

    # Малюємо графік
    plt.subplot(1, 3, idx + 1)
    plt.plot(x_plot, y_true, 'k-', label='Точна')
    plt.plot(x_plot, y_plot, 'r--', label='Інтерполяція')
    plt.scatter(x_test, y_test, color='blue', s=30)
    plt.title(f'[0,{interval}], {n_nodes} вузлів\nПохибка={error:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

plt.suptitle('Дослідження впливу кількості вузлів (фіксований крок h=200)')
plt.tight_layout()
plt.savefig('fixed_step_influence.png', dpi=300, bbox_inches='tight')
plt.show()


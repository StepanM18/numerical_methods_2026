
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import csv
import os


# ============================================================================
# Частина 1: Функції для зчитування даних з файлу
# ============================================================================

def read_csv_file(filename: str) -> Tuple[np.ndarray, np.ndarray]:

    x_list = []
    y_list = []
    line_num = 0

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)

            # Читаємо заголовок
            try:
                header = next(reader)
                line_num += 1
                print(f"Заголовок файлу: {header}")
            except StopIteration:
                raise ValueError("Файл порожній")

            # Читаємо дані
            for row in reader:
                line_num += 1

                # Пропускаємо порожні рядки
                if not row or all(cell.strip() == '' for cell in row):
                    continue

                # Перевіряємо, чи рядок має потрібну кількість елементів
                if len(row) < 2:
                    print(f"Попередження: рядок {line_num} має менше 2-х елементів, пропускаємо")
                    continue

                try:
                    # Спроба конвертувати дані в числа
                    month = float(row[0].strip())
                    temp = float(row[1].strip())

                    x_list.append(month)
                    y_list.append(temp)

                except ValueError as e:
                    print(f"Попередження: рядок {line_num} містить нечислові дані: {row}")
                    continue

        # Перевіряємо, чи були зчитані дані
        if len(x_list) == 0:
            raise ValueError("Не вдалося зчитати жодного рядка з даними")

        # Конвертуємо в numpy масиви
        x = np.array(x_list)
        y = np.array(y_list)


        print(f"Зчитано {len(x)} точок з файлу {filename}")
        print(f"Діапазон місяців: від {x.min()} до {x.max()}")
        print(f"Діапазон температур: від {y.min():.1f} до {y.max():.1f}")

        return x, y


    except Exception as e:
        print(f"ПОМИЛКА при читанні файлу: {e}")
        raise


# ============================================================================
# Частина 2: Функції для методу найменших квадратів
# ============================================================================

def form_matrix(x: np.ndarray, m: int) -> np.ndarray:

    n = len(x)
    matrix_size = m + 1
    A = np.zeros((matrix_size, matrix_size))

    for k in range(matrix_size):
        for l in range(matrix_size):
            # Обчислюємо b_kl = sum(x_i^(k+l))
            A[k, l] = np.sum(x ** (k + l))

    return A


def form_vector(x: np.ndarray, y: np.ndarray, m: int) -> np.ndarray:

    vector_size = m + 1
    b = np.zeros(vector_size)

    for k in range(vector_size):
        # Обчислюємо c_k = sum(y_i * x_i^k)
        b[k] = np.sum(y * (x ** k))

    return b


# ============================================================================
# Частина 3: Розв'язування СЛАР методом Гауса з вибором головного елемента
# ============================================================================

def gauss_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:

    n = len(b)
    # Створюємо копії, щоб не змінювати оригінали
    A = A.copy().astype(float)
    b = b.copy().astype(float)

    # Прямий хід методу Гауса з вибором головного елемента
    for k in range(n):
        # 1. Вибір головного елемента в k-му стовпчику
        max_row = k
        max_val = abs(A[k, k])

        for i in range(k + 1, n):
            if abs(A[i, k]) > max_val:
                max_val = abs(A[i, k])
                max_row = i

        # Перевірка на виродженість матриці
        if max_val < 1e-15:
            raise ValueError(f"Матриця вироджена: головний елемент близький до нуля на кроці {k}")

        # 2. Перестановка рядків, якщо потрібно
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
            print(f"  Перестановка рядків {k} та {max_row}")

        # 3. Виключення елементів під головним елементом
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотній хід методу Гауса
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ax = np.dot(A[i, i + 1:], x[i + 1:])
        x[i] = (b[i] - sum_ax) / A[i, i]

    return x


# ============================================================================
# Частина 4: Функції для обчислення многочлена та похибок
# ============================================================================

def polynomial(x: np.ndarray, coef: np.ndarray) -> np.ndarray:

    result = np.zeros_like(x, dtype=float)

    for i, a in enumerate(coef):
        result += a * (x ** i)

    return result


def compute_variance(y_true: np.ndarray, y_approx: np.ndarray) -> float:

    n = len(y_true)
    squared_errors = (y_approx - y_true) ** 2
    variance = np.sqrt(np.sum(squared_errors) / (n))
    return variance


def compute_error(y_true: np.ndarray, y_approx: np.ndarray) -> np.ndarray:

    return np.abs(y_true - y_approx)


# ============================================================================
# Частина 5: Табуляція даних
# ============================================================================

def tabulate_data(x: np.ndarray, y: np.ndarray,
                  x0: float, xn: float, n_points: int) -> Tuple[np.ndarray, np.ndarray]:

    # Створюємо рівномірну сітку
    x_tab = np.linspace(x0, xn, n_points)

    # Інтерполюємо значення (лінійна інтерполяція)
    y_tab = np.interp(x_tab, x, y)

    return x_tab, y_tab


# ============================================================================
# Частина 6: Основна програма
# ============================================================================

def main():

    print("=" * 70)
    print("ЛАБОРАТОРНА РОБОТА №4")
    print("Метод найменших квадратів для апроксимації многочленами")
    print("=" * 70)

    # ========================================================================
    # Завдання 1: Зчитування даних з CSV файлу
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЗАВДАННЯ 1: Зчитування даних з CSV файлу")
    print("=" * 70)

    filename = "temperatures.csv"  # Назва файлу з даними

    try:
        x, y = read_csv_file(filename)
    except Exception as e:
        print(f"Критична помилка: {e}")
        print("Програма завершує роботу.")
        return

    n = len(x) - 1  # n = 23 (індекси від 0 до n)
    print(f"\nЗчитано {n + 1} точок (місяців)")
    print(f"x0 = {x[0]}, xn = {x[n]}")

    # ========================================================================
    # Завдання 1 (продовження): Табуляція даних
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЗАВДАННЯ 1 (продовження): Табуляція даних")
    print("=" * 70)

    x0, xn = x[0], x[n]
    h = (xn - x0) / n  # крок для табуляції
    n_tab = 24  # кількість точок для табуляції (згідно з завданням)

    x_tab, y_tab = tabulate_data(x, y, x0, xn, n_tab)

    print(f"Табуляція на відрізку [{x0}, {xn}] з кроком h = {h:.2f}")
    print(f"Отримано {len(x_tab)} точок")

    # Виведемо перші 5 точок для перевірки
    print("\nПерші 5 точок після табуляції:")
    for i in range(min(5, len(x_tab))):
        print(f"  x[{i}] = {x_tab[i]:.2f}, y[{i}] = {y_tab[i]:.2f}")

    # ========================================================================
    # Завдання 2: Пошук апроксимуючих многочленів для різних степенів m
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЗАВДАННЯ 2-3: Пошук апроксимуючих многочленів для m = 1..10")
    print("=" * 70)

    max_degree = 10
    variances = []
    coefficients = []  # зберігаємо коефіцієнти для кожного m

    for m in range(1, max_degree + 1):
        print(f"\n--- Апроксимація многочленом степеня m = {m} ---")

        # Формуємо систему нормальних рівнянь
        A = form_matrix(x_tab, m)
        b = form_vector(x_tab, y_tab, m)

        print(f"  Матриця A розміром {A.shape}")
        print(f"  Вектор b розміром {len(b)}")

        # Розв'язуємо систему
        try:
            coef = gauss_solve(A, b)
            coefficients.append(coef)

            # Обчислюємо наближені значення
            y_approx = polynomial(x_tab, coef)

            # Обчислюємо дисперсію
            var = compute_variance(y_tab, y_approx)
            variances.append(var)

            print(f"  Коефіцієнти многочлена: {coef}")
            print(f"  Дисперсія δ = {var:.6f}")

        except Exception as e:
            print(f"  ПОМИЛКА: {e}")
            variances.append(float('inf'))
            coefficients.append(None)

    # ========================================================================
    # Завдання 3: Вибір оптимального степеня многочлена
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЗАВДАННЯ 3: Вибір оптимального степеня многочлена")
    print("=" * 70)

    # Знаходимо степінь з мінімальною дисперсією
    valid_variances = [v for v in variances if v != float('inf')]
    if valid_variances:
        optimal_m = np.argmin(variances[:len(valid_variances)]) + 1
        optimal_var = variances[optimal_m - 1]

        print(f"\nРезультати для всіх степенів m:")
        for m in range(1, max_degree + 1):
            status = ";" if variances[m - 1] != float('inf') else "✗"
            print(f"  m = {m}: δ = {variances[m - 1]:.6f} {status}")

        print(f"\nОптимальний степінь многочлена: m_opt = {optimal_m}")
        print(f"Мінімальна дисперсія: δ_min = {optimal_var:.6f}")
    else:
        print("Не вдалося обчислити жодної дисперсії")
        return

    # ========================================================================
    # Завдання 4: Побудова графіка залежності дисперсії від степеня
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЗАВДАННЯ 4: Побудова графіка залежності дисперсії від степеня")
    print("=" * 70)

    plt.figure(figsize=(12, 8))

    # Графік залежності дисперсії від степеня многочлена
    plt.subplot(2, 2, 1)
    m_values = list(range(1, max_degree + 1))
    plt.plot(m_values, variances, 'bo-', linewidth=2, markersize=8)
    plt.plot(optimal_m, optimal_var, 'r*', markersize=15, label=f'Оптимум: m={optimal_m}')
    plt.xlabel('Степінь многочлена m')
    plt.ylabel('Дисперсія δ')
    plt.title('Залежність дисперсії від степеня многочлена')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')  # логарифмічна шкала для кращої візуалізації

    # ========================================================================
    # Завдання 5: Апроксимація оптимальним многочленом
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЗАВДАННЯ 5: Апроксимація оптимальним многочленом")
    print("=" * 70)

    # Отримуємо коефіцієнти оптимального многочлена
    opt_coef = coefficients[optimal_m - 1]

    # Обчислюємо значення оптимального многочлена
    x_detailed = np.linspace(x0, xn, 200)  # детальна сітка для гладкого графіка
    y_opt_approx = polynomial(x_detailed, opt_coef)
    y_opt_at_points = polynomial(x_tab, opt_coef)

    # Обчислюємо похибку
    error = compute_error(y_tab, y_opt_at_points)

    print(f"\nКоефіцієнти оптимального многочлена степені {optimal_m}:")
    for i, a in enumerate(opt_coef):
        print(f"  a_{i} = {a:.6f}")

    print(f"\nДисперсія оптимального многочлена: δ = {optimal_var:.6f}")
    print(f"Середня похибка: {np.mean(error):.6f}")
    print(f"Максимальна похибка: {np.max(error):.6f}")

    # ========================================================================
    # Завдання 5 (продовження): Графік апроксимації
    # ========================================================================
    plt.subplot(2, 2, 2)
    plt.plot(x, y, 'ro', label='Фактичні дані', markersize=6)
    plt.plot(x_tab, y_tab, 'bo', label='Табульовані дані', markersize=4)
    plt.plot(x_detailed, y_opt_approx, 'g-', linewidth=2,
             label=f'Апроксимація (m={optimal_m})')
    plt.xlabel('Місяць')
    plt.ylabel('Температура (°C)')
    plt.title('Апроксимація температури методом найменших квадратів')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # ========================================================================
    # Завдання 5 (продовження): Графік похибки
    # ========================================================================
    plt.subplot(2, 2, 3)
    plt.bar(x_tab, error, width=0.7, color='orange', alpha=0.7)
    plt.xlabel('Місяць')
    plt.ylabel('Похибка |f(x) - φ(x)|')
    plt.title('Похибка апроксимації оптимальним многочленом')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=np.mean(error), color='r', linestyle='--',
                label=f'Середня похибка: {np.mean(error):.3f}')
    plt.legend()

    # ========================================================================
    # Завдання 6: Прогноз на наступні 3 місяці
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЗАВДАННЯ 6: Прогноз температури на наступні 3 місяці")
    print("=" * 70)

    forecast_m = 3

    # Будуємо многочлен для прогнозу
    A_forecast = form_matrix(x_tab, forecast_m)
    b_forecast = form_vector(x_tab, y_tab, forecast_m)
    coef_forecast = gauss_solve(A_forecast, b_forecast)

    # Прогноз на наступні 3 місяці
    future_months = np.array([25, 26, 27])
    future_temps = polynomial(future_months, coef_forecast)
    #future_temps = polynomial(future_months, opt_coef)

    print("\nПрогноз температури:")
    for month, temp in zip(future_months, future_temps):
        print(f"  Місяць {int(month)}: {temp:.2f}°C")

    # Додаємо прогноз на графік
    plt.subplot(2, 2, 4)
    # Відображаємо всі дані + прогноз
    all_x = np.concatenate([x, future_months])
    all_y_actual = np.concatenate([y, [np.nan, np.nan, np.nan]])
    all_y_pred = np.concatenate([y, future_temps])

    plt.plot(x, y, 'ro-', label='Історичні дані', linewidth=1, markersize=6)
    plt.plot(future_months, future_temps, 'gs--', label='Прогноз',
             linewidth=2, markersize=8)
    plt.xlabel('Місяць')
    plt.ylabel('Температура (°C)')
    plt.title('Прогноз температури на 3 місяці')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('lab4_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ========================================================================
    # Завдання 7: Додатковий аналіз - похибка для різних m
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЗАВДАННЯ 7: Детальний аналіз похибок для різних m")
    print("=" * 70)

    # Табуляція похибки на відрізку [x0, xn] з кроком h1 = (xn-x0)/(20*n)
    h1 = (xn - x0) / (20 * n)
    x_error_detailed = np.arange(x0, xn + h1, h1)

    plt.figure(figsize=(14, 10))

    # Покажемо похибки для кількох значень m

    m_to_show = list(range(1, 11))
    colors = plt.cm.viridis(np.linspace(0, 1, len(m_to_show)))

    for idx, m in enumerate(m_to_show):
        if m <= max_degree and coefficients[m - 1] is not None:
            coef_m = coefficients[m - 1]
            y_m_approx = polynomial(x_error_detailed, coef_m)
            y_m_at_points = np.interp(x_error_detailed, x_tab, y_tab)
            error_m = np.abs(y_m_at_points - y_m_approx)

            plt.plot(x_error_detailed, error_m, color=colors[idx],
                     linewidth=2, label=f'm = {m}')

    plt.xlabel('Місяць')
    plt.ylabel('Похибка ε(x)')
    plt.title('Похибка апроксимації для різних степенів многочлена')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')  # логарифмічна шкала для кращої візуалізації
    plt.tight_layout()
    plt.savefig('lab4_errors.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nПерші 10 точок табуляції похибки (крок h={h1:.6f}):")
    print(
     f"{'x':>6} | {'m=1':>8} {'m=2':>8} {'m=3':>8} {'m=4':>8} {'m=5':>8} {'m=6':>8} {'m=7':>8} {'m=8':>8} {'m=9':>8} {'m=10':>8}")
    print("-" * 106)

    for i in range(min(10, len(x_error_detailed))):
        x_val = x_error_detailed[i]
        print(f"{x_val:6.2f} | ", end="")
        for idx, m in enumerate(m_to_show):
            if m <= max_degree and coefficients[m - 1] is not None:
                coef_m = coefficients[m - 1]
                y_m_approx = polynomial(np.array([x_val]), coef_m)[0]
                y_true_at_x = np.interp([x_val], x_tab, y_tab)[0]
                error = abs(y_true_at_x - y_m_approx)
                print(f"{error:8.4f} ", end="")
        print()

    # ========================================================================
    # Виведення підсумків
    # ========================================================================
    print("\n" + "=" * 70)
    print("ПІДСУМКИ ЛАБОРАТОРНОЇ РОБОТИ")
    print("=" * 70)
    print(f" Зчитано дані з файлу {filename}: {len(x)} точок")
    print(f" Виконано табуляцію на відрізку [{x0}, {xn}]")
    print(f" Побудовано апроксимуючі многочлени для m = 1..{max_degree}")
    print(f" Оптимальний степінь многочлена: m_opt = {optimal_m}")
    print(f" Мінімальна дисперсія: δ_min = {optimal_var:.6f}")
    print(f" Побудовано графіки апроксимації та похибок")
    print(f" Виконано прогноз на 3 місяці: {future_temps[0]:.1f}°C, {future_temps[1]:.1f}°C, {future_temps[2]:.1f}°C")
    print("=" * 70)


# ============================================================================
# Запуск програми
# ============================================================================
if __name__ == "__main__":
    main()
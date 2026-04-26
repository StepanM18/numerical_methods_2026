import random
import math


# ============================================================
# ЗАВДАННЯ 1: Генерація матриці A з діагональним переважанням
# та вектора B
# ============================================================

def generate_matrix_with_diagonal_dominance(n, filename="matrix_A.txt"):
    """
    Генерує матрицю A розмірності n x n з діагональним переважанням.
    Діагональне переважання: |a_ii| > сума |a_ij| для j≠i
    Це забезпечує збіжність ітераційних методів (згідно з методичкою).
    """
    A = []
    for i in range(n):
        row = []
        # Генеруємо випадкові числа для рядка (від -5 до 5)
        for j in range(n):
            row.append(random.uniform(-5, 5))
        # Обчислюємо суму модулів всіх елементів рядка
        sum_abs = sum(abs(x) for x in row)
        # Робимо діагональний елемент більшим за суму інших
        # Це гарантує діагональне переважання
        row[i] = sum_abs + random.uniform(1, 10)
        A.append(row)

    # Записуємо матрицю у файл
    with open(filename, "w") as f:
        for row in A:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")

    return A


def compute_B(A, X_exact):
    """
    Обчислює вектор B = A * X_exact.
    X_exact - точний розв'язок (задаємо всі x = 2.5).
    """
    n = len(A)
    B = [0.0] * n
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += A[i][j] * X_exact[j]
        B[i] = total
    return B


def save_vector_B(B, filename="vector_B.txt"):
    """Зберігає вектор B у текстовий файл."""
    with open(filename, "w") as f:
        f.write(" ".join(f"{val:.6f}" for val in B) + "\n")


# ============================================================
# ЗАВДАННЯ 2: Функції для читання даних
# ============================================================

def read_matrix_A(filename="matrix_A.txt"):
    """Зчитує матрицю A з текстового файлу."""
    A = []
    with open(filename, "r") as f:
        for line in f:
            row = list(map(float, line.strip().split()))
            A.append(row)
    return A


def read_vector_B(filename="vector_B.txt"):
    """Зчитує вектор B з текстового файлу."""
    with open(filename, "r") as f:
        B = list(map(float, f.readline().strip().split()))
    return B


# ============================================================
# Допоміжні функції (норма, множення матриці на вектор)
# ============================================================

def vector_norm(vec):
    """
    Обчислює норму вектора: максимальний елемент за модулем.
    Використовується для перевірки збіжності.
    """
    return max(abs(v) for v in vec)


def matrix_vector_multiply(A, X):
    """
    Обчислює добуток матриці A на вектор X.
    Повертає вектор Y = A * X.
    """
    n = len(A)
    Y = [0.0] * n
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += A[i][j] * X[j]
        Y[i] = total
    return Y


def matrix_norm_inf(C):
    """
    Обчислює норму матриці: максимальна сума модулів по рядках.
    Використовується для перевірки умови збіжності методу простої ітерації.
    """
    n = len(C)
    max_sum = 0.0
    for i in range(n):
        row_sum = sum(abs(C[i][j]) for j in range(n))
        if row_sum > max_sum:
            max_sum = row_sum
    return max_sum


# ============================================================
# МЕТОД 1: ПРОСТА ІТЕРАЦІЯ
# Формула: X^(k+1) = (E - τA)X^(k) + τf
# ============================================================

def simple_iteration_method(A, B, X0, tau, eps=1e-14, max_iter=100000):
    """
    Метод простої ітерації.

    Параметри:
    - A: матриця системи
    - B: вектор правих частин
    - X0: початкове наближення
    - tau: параметр методу (0 < tau < 2/||A||)
    - eps: точність
    - max_iter: максимальна кількість ітерацій

    Повертає: (розв'язок, кількість ітерацій)
    """
    n = len(A)
    X = X0[:]  # копіюємо початкове наближення
    iteration = 0

    while iteration < max_iter:
        iteration += 1

        # Обчислюємо AX^(k)
        AX = matrix_vector_multiply(A, X)

        # Обчислюємо X^(k+1) = X^(k) - τ(AX^(k) - B)
        X_new = [0.0] * n
        for i in range(n):
            X_new[i] = X[i] - tau * (AX[i] - B[i])

        # Перевіряємо умову закінчення: ||X^(k+1) - X^(k)|| < eps
        diff_norm = vector_norm([X_new[i] - X[i] for i in range(n)])

        if diff_norm < eps:
            return X_new, iteration

        X = X_new

    print(f"Метод простої ітерації: досягнуто максимуму ітерацій ({max_iter})")
    return X, iteration


# ============================================================
# МЕТОД 2: ЯКОБІ
# Формула: x_i^(k+1) = - Σ (a_ij / a_ii) * x_j^(k) + f_i / a_ii, де j≠i
# ============================================================

def jacobi_method(A, B, X0, eps=1e-14, max_iter=100000):
    """
    Метод Якобі (паралельних ітерацій).

    Особливість: для обчислення x_i^(k+1) використовуються ТІЛЬКИ
    значення з попередньої ітерації x_j^(k).
    """
    n = len(A)
    X = X0[:]  # поточне наближення
    iteration = 0

    # Попередньо обчислюємо коефіцієнти, щоб не рахувати їх на кожній ітерації
    # Зберігаємо 1/a_ii та a_ij/a_ii для j≠i
    inv_diag = [0.0] * n  # 1 / a_ii
    coeff = []  # список списків: для кожного i список (j, a_ij/a_ii) для j≠i

    for i in range(n):
        inv_diag[i] = 1.0 / A[i][i]
        row_coeff = []
        for j in range(n):
            if j != i:
                row_coeff.append((j, A[i][j] / A[i][i]))
        coeff.append(row_coeff)

    while iteration < max_iter:
        iteration += 1

        # Обчислюємо X^(k+1)
        X_new = [0.0] * n
        for i in range(n):
            # Сума по j≠i: (a_ij / a_ii) * x_j^(k)
            s = 0.0
            for j, coef in coeff[i]:
                s += coef * X[j]
            # x_i^(k+1) = -s + f_i / a_ii
            X_new[i] = -s + B[i] * inv_diag[i]

        # Перевіряємо збіжність
        diff_norm = vector_norm([X_new[i] - X[i] for i in range(n)])

        if diff_norm < eps:
            return X_new, iteration

        X = X_new

    print(f"Метод Якобі: досягнуто максимуму ітерацій ({max_iter})")
    return X, iteration


# ============================================================
# МЕТОД 3: ЗЕЙДЕЛЯ
# Формула: x_i^(k+1) = f_i/a_ii - Σ(j<i) a_ij/a_ii * x_j^(k+1) - Σ(j>i) a_ij/a_ii * x_j^(k)
# ============================================================

def seidel_method(A, B, X0, eps=1e-14, max_iter=100000):
    """
    Метод Зейделя (послідовних ітерацій).

    Особливість: для обчислення x_i^(k+1) використовуються
    ВЖЕ ОНОВЛЕНІ значення x_j^(k+1) для j < i.
    Це прискорює збіжність порівняно з методом Якобі.
    """
    n = len(A)
    X = X0[:]  # поточне наближення
    iteration = 0

    # Попередньо обчислюємо 1/a_ii та коефіцієнти
    inv_diag = [1.0 / A[i][i] for i in range(n)]

    while iteration < max_iter:
        iteration += 1

        # Оновлюємо X покомпонентно, ВИКОРИСТОВУЮЧИ ВЖЕ ОНОВЛЕНІ
        X_new = X[:]  # копіюємо, будемо оновлювати по черзі

        for i in range(n):
            # Сума по j < i (використовуємо x_j^(k+1) - вже оновлені)
            sum1 = 0.0
            for j in range(i):
                sum1 += A[i][j] * X_new[j]

            # Сума по j > i (використовуємо x_j^(k) - старі значення)
            sum2 = 0.0
            for j in range(i + 1, n):
                sum2 += A[i][j] * X[j]

            # x_i^(k+1) = (f_i - sum1 - sum2) / a_ii
            X_new[i] = (B[i] - sum1 - sum2) * inv_diag[i]

        # Перевіряємо збіжність
        diff_norm = vector_norm([X_new[i] - X[i] for i in range(n)])

        if diff_norm < eps:
            return X_new, iteration

        X = X_new

    print(f"Метод Зейделя: досягнуто максимуму ітерацій ({max_iter})")
    return X, iteration


# ============================================================
# Додаткова функція: оцінка точності розв'язку
# Обчислює нев'язку ||AX - B||
# ============================================================

def check_accuracy(A, X, B):
    """Обчислює норму нев'язки: max |AX - B|"""
    AX = matrix_vector_multiply(A, X)
    residual = [AX[i] - B[i] for i in range(len(B))]
    return vector_norm(residual)


# ============================================================
# ГОЛОВНА ПРОГРАМА
# ============================================================

def main():
    # Розмірність системи (згідно з методичкою: n=100)
    n = 100

    # Точний розв'язок (задаємо всі x_i = 2.5)
    X_exact = [2.5] * n

    print("=" * 60)
    print("ЛАБОРАТОРНА РОБОТА №8: ІТЕРАЦІЙНІ МЕТОДИ")
    print("=" * 60)

    # ==========================================================
    # ЗАВДАННЯ 1: Генерація матриці A та вектора B
    # ==========================================================
    print("\n[Крок 1] Генерація матриці A з діагональним переважанням...")
    A = generate_matrix_with_diagonal_dominance(n, "matrix_A.txt")
    print(f"Матрицю A ({n}x{n}) збережено у файл 'matrix_A.txt'")

    print("Обчислення вектора B = A * X_exact (X_exact всі = 2.5)...")
    B = compute_B(A, X_exact)
    save_vector_B(B, "vector_B.txt")
    print("Вектор B збережено у файл 'vector_B.txt'")

    # ==========================================================
    # ЗАВДАННЯ 2-3: Читання даних та початкове наближення
    # ==========================================================
    print("\n[Крок 2-3] Читання даних та задання початкового наближення...")
    A = read_matrix_A("matrix_A.txt")
    B = read_vector_B("vector_B.txt")

    # Початкове наближення: всі x_i = 1.0 (згідно з методичкою)
    X0 = [1.0] * n
    print(f"Початкове наближення X0: всі елементи = 1.0")

    # ==========================================================
    # ЗАВДАННЯ 4: Розв'язання трьома методами
    # ==========================================================

    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 4: РОЗВ'ЯЗАННЯ СЛАР ІТЕРАЦІЙНИМИ МЕТОДАМИ")
    print("=" * 60)

    eps = 1e-14  # задана точність (згідно з методичкою)

    # ----------------------------------------------------------
    # МЕТОД 1: ПРОСТА ІТЕРАЦІЯ
    # ----------------------------------------------------------
    print("\n--- МЕТОД ПРОСТОЇ ІТЕРАЦІЇ ---")

    # Підбираємо параметр tau: 0 < tau < 2/||A||
    # Спочатку обчислимо норму матриці A
    normA = 0.0
    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n))
        if row_sum > normA:
            normA = row_sum

    tau = 1.0 / normA  # вибираємо tau = 1/||A|| (гарантує збіжність)
    print(f"Норма матриці A = {normA:.6f}")
    print(f"Параметр tau = {tau:.6f} (0 < tau < {2 / normA:.6f})")

    X_simple, iter_simple = simple_iteration_method(A, B, X0, tau, eps)
    res_simple = check_accuracy(A, X_simple, B)
    print(f"Кількість ітерацій: {iter_simple}")
    print(f"Нев'язка ||AX - B|| = {res_simple:.2e}")
    print(f"Перші 5 компонент розв'язку: {[round(x, 6) for x in X_simple[:5]]}")

    # ----------------------------------------------------------
    # МЕТОД 2: ЯКОБІ
    # ----------------------------------------------------------
    print("\n--- МЕТОД ЯКОБІ ---")
    X_jacobi, iter_jacobi = jacobi_method(A, B, X0, eps)
    res_jacobi = check_accuracy(A, X_jacobi, B)
    print(f"Кількість ітерацій: {iter_jacobi}")
    print(f"Нев'язка ||AX - B|| = {res_jacobi:.2e}")
    print(f"Перші 5 компонент розв'язку: {[round(x, 6) for x in X_jacobi[:5]]}")

    # ----------------------------------------------------------
    # МЕТОД 3: ЗЕЙДЕЛЯ
    # ----------------------------------------------------------
    print("\n--- МЕТОД ЗЕЙДЕЛЯ ---")
    X_seidel, iter_seidel = seidel_method(A, B, X0, eps)
    res_seidel = check_accuracy(A, X_seidel, B)
    print(f"Кількість ітерацій: {iter_seidel}")
    print(f"Нев'язка ||AX - B|| = {res_seidel:.2e}")
    print(f"Перші 5 компонент розв'язку: {[round(x, 6) for x in X_seidel[:5]]}")

    # ==========================================================
    # ПОРІВНЯННЯ МЕТОДІВ
    # ==========================================================
    print("\n" + "=" * 60)
    print("ПОРІВНЯННЯ МЕТОДІВ")
    print("=" * 60)
    print(f"{'Метод':<20} {'Ітерації':<15} {'Нев\'язка':<15}")
    print("-" * 50)
    print(f"{'Проста ітерація':<20} {iter_simple:<15} {res_simple:.2e}")
    print(f"{'Якобі':<20} {iter_jacobi:<15} {res_jacobi:.2e}")
    print(f"{'Зейделя':<20} {iter_seidel:<15} {res_seidel:.2e}")

    # Перевірка: чи збігається з точним розв'язком?
    print("\n" + "-" * 50)
    print("Перевірка: порівняння з точним розв'язком (всі 2.5)")
    print(f"Похибка методу Зейделя: {max(abs(X_seidel[i] - 2.5) for i in range(n)):.2e}")




if __name__ == "__main__":
    main()
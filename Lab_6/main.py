import random
import math

# ============================================================
# 1. Генерація матриці A, задання точного розв'язку X_exact,
#    обчислення вектора B, збереження у файли.
# ============================================================

def generate_and_save_matrix_A(n, filename="matrix_A.txt"):
    """
    Генерує випадкову матрицю A розмірності n x n.
    Елементи - випадкові дійсні числа від -10.0 до 10.0.
    """
    A = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
    with open(filename, "w") as f:
        for row in A:
            f.write(" ".join(f"{val:.3f}" for val in row) + "\n")
    return A

def save_vector_B(B, filename="vector_B.txt"):
    """Зберігає вектор B у текстовий файл."""
    with open(filename, "w") as f:
        f.write(" ".join(f"{val:.3f}" for val in B) + "\n")

def compute_B(A, X_exact):
    """
    Обчислює B = A * X_exact.
    """
    n = len(A)
    B = [0.0] * n
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += A[i][j] * X_exact[j]
        B[i] = total
    return B

# ============================================================
# 2. Функції для читання A та B з файлів
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
# 3. LU-розклад (згідно з формулами з методички)
# ============================================================

def lu_decomposition(A):
    """
    Виконує LU-розклад матриці A.
    L - нижня трикутна, U - верхня трикутна з одиницями на діагоналі.
    Повертає (L, U).
    """
    n = len(A)
    # Ініціалізуємо L та U нулями
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    # Крок 2 з методички: діагональні елементи U = 1
    for i in range(n):
        U[i][i] = 1.0

    # Крок 3: почергово обчислюємо стовпці L та рядки U
    for k in range(n):
        # Обчислюємо k-й стовпець L (для i >= k)
        for i in range(k, n):
            s = 0.0
            for j in range(k):
                s += L[i][j] * U[j][k]
            L[i][k] = A[i][k] - s

        # Обчислюємо k-й рядок U (для елементів правіше діагоналі)
        for i in range(k + 1, n):
            s = 0.0
            for j in range(k):
                s += L[k][j] * U[j][i]
            U[k][i] = (A[k][i] - s) / L[k][k]

    return L, U

def save_LU(L, U, filename="LU_decomposition.txt"):
    """Зберігає матриці L та U у файл."""
    with open(filename, "w") as f:
        f.write("Matrix L (lower triangular):\n")
        for row in L:
            f.write(" ".join(f"{val:.3f}" for val in row) + "\n")
        f.write("\nMatrix U (upper triangular with ones on diagonal):\n")
        for row in U:
            f.write(" ".join(f"{val:.3f}" for val in row) + "\n")

# ============================================================
# 4. Розв'язання СЛАР за допомогою LU-розкладу
#    (L*Z = B, потім U*X = Z)
# ============================================================

def solve_lu(L, U, B):
    """
    Розв'язує систему A*X = B, де A = L*U.
    """
    n = len(L)
    Z = [0.0] * n

    # Прямий хід: L * Z = B
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i][j] * Z[j]
        Z[i] = (B[i] - s) / L[i][i]

    # Зворотний хід: U * X = Z
    X = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i][j] * X[j]
        X[i] = Z[i] - s   # бо U[i][i] = 1

    return X

def save_solution(X, filename="solution.txt"):
    """Зберігає знайдений розв'язок X у файл."""
    with open(filename, "w") as f:
        f.write(" ".join(f"{val:.15f}" for val in X) + "\n")

# ============================================================
# 5. Обчислення норми вектора (максимум модуля)
# ============================================================

def vector_norm(vec):
    """Повертає норму вектора: max |vec[i]|."""
    return max(abs(v) for v in vec)

# ============================================================
# 6. Обчислення добутку матриці на вектор
# ============================================================

def mat_vec_mul(A, X):
    """Повертає вектор Y = A * X."""
    n = len(A)
    Y = [0.0] * n
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += A[i][j] * X[j]
        Y[i] = total
    return Y

# ============================================================
# 7. Ітераційне уточнення розв'язку (згідно з методичкою)
# ============================================================

def iterative_refinement(A, L, U, B, X0, eps=1e-14, max_iter=100):
    """
    Уточнює розв'язок X0 за допомогою методу з методички:
    - обчислюємо нев'язку R = B - A*X0
    - розв'язуємо A * dX = R  (через LU)
    - X_new = X0 + dX
    - перевіряємо ||dX|| <= eps та ||A*X_new - B|| <= eps
    """
    iteration = 0
    X = X0[:] # створює повну копію списку
    while iteration < max_iter:
        # Обчислюємо вектор нев'язки R = B - A*X
        AX = mat_vec_mul(A, X)
        R = [B[i] - AX[i] for i in range(len(B))]

        # Розв'язуємо A * dX = R (використовуємо LU-розклад)
        dX = solve_lu(L, U, R)

        # Обчислюємо норму похибки
        norm_dX = vector_norm(dX)

        # Уточнюємо розв'язок
        X_new = [X[i] + dX[i] for i in range(len(X))]

        # Перевіряємо умови закінчення
        AX_new = mat_vec_mul(A, X_new)
        residual_new = [B[i] - AX_new[i] for i in range(len(B))]
        norm_residual = vector_norm(residual_new)

        if norm_dX <= eps and norm_residual <= eps:
            print(f"Уточнення зійшлося за {iteration + 1} ітерацій.")
            return X_new, iteration + 1

        X = X_new
        iteration += 1

    print(f"Досягнуто максимуму ітерацій ({max_iter}).")
    return X, iteration

# ============================================================
# ГОЛОВНА ПРОГРАМА
# ============================================================

def main():
    # Розмірність системи
    n = 100

    # Точний розв'язок (задаємо довільно, наприклад, усі xi = 2.5)
    X_exact = [2.5] * n

    print("Крок 1: Генерація матриці A та вектора B...")
    A = generate_and_save_matrix_A(n, "matrix_A.txt")
    B = compute_B(A, X_exact)
    save_vector_B(B, "vector_B.txt")
    print("Матрицю A збережено у matrix_A.txt")
    print("Вектор B збережено у vector_B.txt")

    print("\nКрок 2: Читання A та B з файлів...")
    A = read_matrix_A("matrix_A.txt")
    B = read_vector_B("vector_B.txt")

    print("\nКрок 3: LU-розклад...")
    L, U = lu_decomposition(A)
    save_LU(L, U, "LU_decomposition.txt")
    print("LU-розклад збережено у LU_decomposition.txt")

    print("\nКрок 4: Розв'язання СЛАР за допомогою LU-розкладу...")
    X = solve_lu(L, U, B)
    save_solution(X, "solution.txt")
    print("Розв'язок збережено у solution.txt")

    print("\nКрок 5: Оцінка точності знайденого розв'язку...")
    AX = mat_vec_mul(A, X)
    eps_accuracy = max(abs(AX[i] - B[i]) for i in range(n))
    print(f"Точність (max |A*X - B|) = {eps_accuracy:.2e}")

    print("\nКрок 6: Ітераційне уточнення розв'язку...")
    X_refined, iterations = iterative_refinement(A, L, U, B, X, eps=1e-14)
    print(f"Уточнений розв'язок (перші 5 компонент): {X_refined[:100]}")
    print(f"Кількість ітерацій: {iterations}")

    # Додаткова перевірка для уточненого розв'язку
    AX_refined = mat_vec_mul(A, X_refined)
    final_residual = max(abs(AX_refined[i] - B[i]) for i in range(n))
    print(f"Нев'язка після уточнення: {final_residual:.2e}")

if __name__ == "__main__":
    main()
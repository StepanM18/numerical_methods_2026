
import math
import numpy as np
import matplotlib.pyplot as plt



# 1. Табуляція трансцендентної функції F(x) на відрізку [a, b] з кроком h

def F_transcendental(x):

    return x * math.exp(-x) - 0.5 * math.sin(x)


def tabulate_function(a, b, h, func, filename="tabulation.txt"):

    x_vals = []
    y_vals = []
    with open(filename, "w", encoding="utf-8") as f:
        x = a
        while x <= b + 1e-12:  # щоб уникнути помилок округлення
            y = func(x)
            x_vals.append(x)
            y_vals.append(y)
            f.write(f"{x:.6f} {y:.6f}\n")
            x += h
    return list(zip(x_vals, y_vals))


def find_initial_roots(tab_data):

    #З отриманої сукупності вузлів {xi, yi} наближено знаходить абсциси точок перетину функції з віссю x (де змінюється знак).
    roots = []
    for i in range(len(tab_data) - 1):
        x1, y1 = tab_data[i]
        x2, y2 = tab_data[i + 1]
        if y1 * y2 < 0:  # зміна знаку
            # лінійна інтерполяція для початкового наближення
            x_root = x1 - y1 * (x2 - x1) / (y2 - y1)
            roots.append(x_root)
    return roots



# 2,3,4. Розв'язок F(x)=0 різними методами

#Проста ітерація
def simple_iteration(F, x0, eps=1e-10, max_iter=1000, tau=0.1):

   # Метод простої ітерації


    x = x0
    history = [x]  # початкове наближення
    for n in range(max_iter):
        Fx = F(x)
        x_next = x + tau * Fx # формула
        history.append(x_next) # додаємо в кінець списку
        # Критерії зупинки:
        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, n + 1, history
        x = x_next
    raise RuntimeError("Метод простої ітерації не зійшовся")

#Ньютона
def newton_method(F, dF, x0, eps=1e-10, max_iter=100):

    #Метод Ньютона (другий порядок збіжності).

    x = x0
    history = [x]
    for n in range(max_iter):
        Fx = F(x)
        dFx = dF(x)
        if abs(dFx) < 1e-15:
            raise RuntimeError("Похідна близька до нуля")
        x_next = x - Fx / dFx #формула
        history.append(x_next)
        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, n + 1, history
        x = x_next
    raise RuntimeError("Метод Ньютона не зійшовся")

#Чебишев
def chebyshev_method(F, dF, d2F, x0, eps=1e-10, max_iter=100):

    #Метод Чебишева

    x = x0
    history = [x]
    for n in range(max_iter):
        Fx = F(x)
        dFx = dF(x)
        d2Fx = d2F(x)
        if abs(dFx) < 1e-15:
            raise RuntimeError("Похідна близька до нуля")
        term1 = Fx / dFx
        term2 = 0.5 * (Fx ** 2 * d2Fx) / (dFx ** 3)
        x_next = x - term1 - term2 #формула
        history.append(x_next)
        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, n + 1, history
        x = x_next
    raise RuntimeError("Метод Чебишева не зійшовся")


#хорд
def chord_method(F, x0, x1, eps=1e-10, max_iter=100):

    #Метод хорд
    x_prev = x0
    x_curr = x1
    history = [x_prev, x_curr]
    for n in range(max_iter):
        F_prev = F(x_prev)
        F_curr = F(x_curr)
        if abs(F_curr - F_prev) < 1e-15:
            raise RuntimeError("Значення функції майже однакові")
        x_next = x_curr - F_curr * (x_curr - x_prev) / (F_curr - F_prev)  # формула
        history.append(x_next)
        if abs(F(x_next)) < eps and abs(x_next - x_curr) < eps:
            return x_next, n + 1, history
        x_prev, x_curr = x_curr, x_next
    raise RuntimeError("Метод хорд не зійшовся")

#парабол
def parabola_method(F, x0, x1, x2, eps=1e-10, max_iter=100):

    #Метод парабол

    xnm2, xnm1, xn = x0, x1, x2
    history = [xnm2, xnm1, xn]
    for n in range(max_iter):
        F0, F1, F2 = F(xnm2), F(xnm1), F(xn)
        # Розділені різниці
        f01 = (F1 - F0) / (xnm1 - xnm2)
        f12 = (F2 - F1) / (xn - xnm1)
        f012 = (f12 - f01) / (xn - xnm2)

        a = f012
        b = f12 + f012 * (xn - xnm1)
        c = F2
        if abs(a) < 1e-15:

            if abs(b) < 1e-15:
                delta = 0
            else:
                delta = -c / b
        else:
            disc = b ** 2 - 4 * a * c
            if disc < 0:
                # Беремо дійсну частину комплексного кореня
                delta = -b / (2 * a)
            else:
                sqrt_disc = math.sqrt(disc)
                delta1 = (-b + sqrt_disc) / (2 * a)
                delta2 = (-b - sqrt_disc) / (2 * a)
                delta = delta1 if abs(delta1) < abs(delta2) else delta2
        x_next = xn + delta # нове наближення
        history.append(x_next)
        if abs(F(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, n + 1, history
        xnm2, xnm1, xn = xnm1, xn, x_next
    raise RuntimeError("Метод парабол не зійшовся")

#зворотної інтерполяції
def inverse_interpolation(F, x0, x1, x2, eps=1e-10, max_iter=100):
    """
    Метод зворотної інтерполяції (за формулою для трьох вузлів).
    x = L2(y), підставляємо y=0.
    """
    xnm2, xnm1, xn = x0, x1, x2
    history = [xnm2, xnm1, xn]
    for n in range(max_iter):
        # Обчислюємо y = F(x) для кожної точки
        y_nm2 = F(xnm2)
        y_nm1 = F(xnm1)
        y_n = F(xn)

        # Перевірка на співпадіння значень (щоб уникнути ділення на нуль)
        if abs(y_nm2 - y_nm1) < 1e-15 or abs(y_nm2 - y_n) < 1e-15 or abs(y_nm1 - y_n) < 1e-15:
            raise RuntimeError("Співпадають значення функції в вузлах")

        # За формулою для трьох вузлів
        term1 = (y_nm1 * y_n) / ((y_nm2 - y_nm1) * (y_nm2 - y_n)) * xnm2
        term2 = (y_nm2 * y_n) / ((y_nm1 - y_nm2) * (y_nm1 - y_n)) * xnm1
        term3 = (y_nm2 * y_nm1) / ((y_n - y_nm2) * (y_n - y_nm1)) * xn
        x_next = term1 + term2 + term3  #  основна формула
        history.append(x_next)
        if abs(F(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, n + 1, history
        xnm2, xnm1, xn = xnm1, xn, x_next
    raise RuntimeError("Метод зворотної інтерполяції не зійшовся")



# 5. Алгебраїчне рівняння третього порядку

#  (x - 1)(x^2 + 1) = x^3 - x^2 + x - 1

def polynomial_coeffs():


    return [-1, 1, -1, 1]


def F_poly(x, coeffs):
    #Значення алгебраїчного многочлена в точці x за схемою Горнера
    res = 0.0
    for a in reversed(coeffs):
        res = res * x + a
    return res


def F_poly_derivative(x, coeffs):
   # Похідна многочлена: обчислюємо аналітично через коефіцієнти
    deg = len(coeffs) - 1
    res = 0.0
    for i in range(1, deg + 1):
        res += i * coeffs[i] * (x ** (i - 1))
    return res



# 6,7. Зчитування коефіцієнтів з файлу

def read_coeffs_from_file(filename="coeffs.txt"):
    """Читає коефіцієнти з текстового файлу (рядок чисел)"""
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        coeffs = list(map(float, line.split()))
    return coeffs



# 8. Метод Ньютона зі схемою Горнера для алгебраїчного рівняння

def horner(x, coeffs):

    #Схема Горнера: повертає (b0, c1), де b0 = F(x), c1 = F'(x)

    m = len(coeffs) - 1  # степінь
    b = [0.0] * (m + 1)
    b[m] = coeffs[m]
    for i in range(m - 1, -1, -1):
        b[i] = coeffs[i] + x * b[i + 1]
    # b[0] = F(x)
    # тепер обчислюємо c1 для похідної
    c = [0.0] * (m + 1)
    c[m] = b[m]
    for i in range(m - 1, 0, -1):
        c[i] = b[i] + x * c[i + 1]
    c1 = b[1] + x * c[2] if m >= 2 else b[1]
    return b[0], c1


def newton_horner(coeffs, x0, eps=1e-10, max_iter=100):

    #Метод Ньютона з використанням схеми Горнера для многочленів.

    x = x0
    history = [x]
    for n in range(max_iter):
        Fx, dFx = horner(x, coeffs)
        if abs(dFx) < 1e-15:
            raise RuntimeError("Похідна близька до нуля")
        x_next = x - Fx / dFx
        history.append(x_next)
        if abs(Fx) < eps and abs(x_next - x) < eps:
            return x_next, n + 1, history
        x = x_next
    raise RuntimeError("Метод Ньютона-Горнера не зійшовся")



# 9. Метод Ліна для знаходження комплексних коренів

def lin_method(coeffs, alpha0, beta0, eps=1e-10, max_iter=100):

   # Метод Ліна для знаходження комплексно-спряжених коренів α ± iβ многочлен ділиться на квадратний тричлен x2 + px + q

    m = len(coeffs) - 1  # степінь многочлена
    alpha, beta = alpha0, beta0
    history_alpha = [alpha]
    history_beta = [beta]
    for iteration in range(max_iter):
        p = -2 * alpha
        q = alpha * alpha + beta * beta
        # Обчислення коефіцієнтів b_i (зворотний хід)
        b = [0.0] * (m + 1)
        b[m] = coeffs[m]
        b[m - 1] = coeffs[m - 1] + p * b[m]
        for i in range(m - 2, 1, -1):
            b[i] = coeffs[i] + p * b[i + 1] + q * b[i + 2]
        b2 = b[2] if m >= 2 else coeffs[2]
        b3 = b[3] if m >= 3 else 0.0
        if abs(b2) < 1e-15:
            raise RuntimeError("b2 близьке до нуля, метод Ліна не стійкий")
        # Рівняння для нових p1, q1
        a1 = coeffs[1] if len(coeffs) > 1 else 0.0
        a0 = coeffs[0]
        q1 = a0 / b2
        p1 = (a1 * b2 - a0 * b3) / (b2 * b2)
        alpha1 = -p1 / 2
        beta1_sq = q1 - alpha1 * alpha1
        if beta1_sq < 0:
            beta1 = 0.0
        else:
            beta1 = math.sqrt(beta1_sq)
        # Перевірка збіжності
        if abs(alpha1 - alpha) < eps and abs(beta1 - beta) < eps:
            return (alpha1, beta1), iteration + 1, history_alpha, history_beta
        alpha, beta = alpha1, beta1
        history_alpha.append(alpha)
        history_beta.append(beta)
    raise RuntimeError("Метод Ліна не зійшовся")




# Головна програма
def main():
    print("=== Лабораторна робота №9 ===")
    print("1. Табуляція трансцендентної функції")
    a, b, h = 0.0, 3.0, 0.1
    tab_data = tabulate_function(a, b, h, F_transcendental, "tabulation.txt")
    print(f"   Табуляцію виконано, збережено у 'tabulation.txt'")

    # Знаходимо початкові наближення коренів
    initial_roots = find_initial_roots(tab_data)
    print(f"   Знайдено наближених коренів: {initial_roots}")
    # Виберемо два корені: один на зростанні, один на спаданні
    root1_approx = initial_roots[0] if len(initial_roots) > 0 else 0.5
    root2_approx = initial_roots[1] if len(initial_roots) > 1 else 2.0
    print(f"   Для уточнення вибрано: x01={root1_approx:.10f}, x02={root2_approx:.10f}")

    # Похідні для трансцендентної функції (для Ньютона, Чебишева)
    def dF(x):
        # похідна від x*exp(-x) - 0.5*sin(x)
        return math.exp(-x) - x * math.exp(-x) - 0.5 * math.cos(x)

    def d2F(x):
        # друга похідна
        return -math.exp(-x) - math.exp(-x) + x * math.exp(-x) + 0.5 * math.sin(x)

    eps = 1e-10
    print("\n2-4. Уточнення коренів різними методами (точність 1e-10)")

    # Функції-обгортки для кожного методу
    def run_simple_iteration(x0):
        return simple_iteration(F_transcendental, x0, eps, tau=0.1)

    def run_newton(x0):
        return newton_method(F_transcendental, dF, x0, eps)

    def run_chebyshev(x0):
        return chebyshev_method(F_transcendental, dF, d2F, x0, eps)

    def run_chord(x0):
        # для хорд потрібна друга точка
        x1 = x0 + 0.1 if x0 < 2 else x0 - 0.1
        return chord_method(F_transcendental, x0, x1, eps)

    def run_parabola(x0):
        # для парабол потрібні три точки
        x1 = x0 + 0.1
        x2 = x0 + 0.2
        return parabola_method(F_transcendental, x0, x1, x2, eps)

    def run_inverse(x0):
        # для зворотної інтерполяції потрібні три точки
        x1 = x0 + 0.1
        x2 = x0 + 0.2
        return inverse_interpolation(F_transcendental, x0, x1, x2, eps)

    methods = [
        ("Метод простої ітерації", run_simple_iteration),
        ("Метод Ньютона", run_newton),
        ("Метод Чебишева", run_chebyshev),
        ("Метод хорд", run_chord),
        ("Метод парабол", run_parabola),
        ("Метод зворотної інтерполяції", run_inverse)
    ]

    for name, method in methods:
        try:
            # Для першого кореня (зростання)
            root, iters, hist = method(root1_approx)
            print(f"{name}: корінь1 = {root:.12f}, ітерацій = {iters}")
        except Exception as e:
            print(f"{name}: помилка для кореня1 - {e}")

        try:
            # Для другого кореня (спадання)
            root, iters, hist = method(root2_approx)
            print(f"{name}: корінь2 = {root:.12f}, ітерацій = {iters}")
        except Exception as e:
            print(f"{name}: помилка для кореня2 - {e}")

    print("\n5. Алгебраїчне рівняння третього порядку")
    coeffs = polynomial_coeffs()
    print(f"   Коефіцієнти (a0..a3): {coeffs}")
    # Побудова графіка
    x_vals = np.linspace(-2, 3, 200)
    y_vals = [F_poly(x, coeffs) for x in x_vals]
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label="x³ - x² + x - 1", linewidth=2)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, alpha=0.7)
    plt.title("Алгебраїчний многочлен 3-го ступеня")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.legend()
    plt.savefig("polynomial_plot.png", dpi=150)
    plt.show()
    print("   Графік збережено у polynomial_plot.png")

    print("\n6-7. Зчитування коефіцієнтів з файлу (coeffs.txt)")


    import os

    # Записуємо правильні коефіцієнти у файл (a0 a1 a2 a3)
    correct_coeffs = [-1, 1, -1, 1]
    with open("coeffs.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, correct_coeffs)))
    print("   Створено файл coeffs.txt з коефіцієнтами: -1 1 -1 1")

    # Зчитуємо коефіцієнти
    coeffs_from_file = read_coeffs_from_file("coeffs.txt")
    print(f"   Зчитано коефіцієнти: {coeffs_from_file}")

    # Перевірка, що коефіцієнти коректні
    if len(coeffs_from_file) != 4:
        print(f"   ПОМИЛКА: Очікувалось 4 коефіцієнти, отримано {len(coeffs_from_file)}")
        print(f"   Використовуємо правильні коефіцієнти замість зчитаних")
        coeffs_from_file = [-1, 1, -1, 1]

    print("\n8. Знаходження дійсного кореня методом Ньютона-Горнера")
    try:
        real_root, iter_horner, hist_horner = newton_horner(coeffs_from_file, 2.0, eps)
        print(f"   Дійсний корінь = {real_root:.12f}")
        print(f"   Кількість ітерацій: {iter_horner}")
        print(f"   Перевірка: F({real_root:.12f}) = {F_poly(real_root, coeffs_from_file):.2e}")

        # Виведемо історію ітерацій
        print("   Історія уточнення кореня:")
        for i, x_val in enumerate(hist_horner):
            print(f"      x{i} = {x_val:.12f}")
    except Exception as e:
        print(f"   Помилка в методі Ньютона-Горнера: {e}")

    print("\n9. Знаходження комплексних коренів методом Ліна")
    try:
        (alpha, beta), iter_lin, hist_a, hist_b = lin_method(coeffs_from_file, 0.0, 1.0, eps)
        print(f"   Комплексні корені: {alpha:.12f} ± i{beta:.12f}")
        print(f"   Кількість ітерацій: {iter_lin}")



        # Виведемо історію ітерацій
        print("   Історія уточнення комплексних коренів:")
        print("      α (дійсна частина)    β (уявна частина)")
        for i in range(min(len(hist_a), 10)):
            print(f"      {hist_a[i]:.12f}    {hist_b[i]:.12f}")

    except Exception as e:
        print(f"   Метод Ліна: {e}")
        print("   Спробуємо інше початкове наближення...")
        try:
            (alpha, beta), iter_lin, hist_a, hist_b = lin_method(coeffs_from_file, 0.5, 0.8, eps)
            print(f"   Комплексні корені: {alpha:.12f} ± i{beta:.12f}")
            print(f"   Кількість ітерацій: {iter_lin}")
        except Exception as e2:
            print(f"   Повторна помилка: {e2}")


if __name__ == "__main__":
    main()
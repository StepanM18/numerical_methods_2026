import numpy as np
import matplotlib.pyplot as plt

#  Аналітичний розв'язок диференціального рівняння

# Функція правої частини ДР: dy/dx = f(x, y)
def f(x, y):
    return y - x

# Точний (аналітичний) розв'язок
def exact_sol(x):
    return 0.5 * np.exp(x) + x + 1

# Параметри
x0 = 0.0
x_end = 1.5
y0 = 1.5
h = 0.1  # Фіксований крок
eps = 1e-5  # Задана точність для автоматичного вибору кроку


# ДОПОМІЖНА ФУНКЦІЯ: Метод Рунге-Кутта 4 порядку для одного кроку

def rk4_step(f, x, y, h):

    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ДОПОМІЖНА ФУНКЦІЯ: Метод Адамса для одного кроку (потребує 2 попередні точки)
def adams_step(f, x_prev, y_prev, x_curr, y_curr, h):

    #Один крок методу Адамса 2-го порядку (прогноз-корекція)

    f_curr = f(x_curr, y_curr)
    f_prev = f(x_prev, y_prev)

    # Прогноз (предиктор)
    y_pred = y_curr + (h / 2.0) * (3 * f_curr - f_prev)

    # Корекція
    x_next = x_curr + h
    y_corr = y_pred
    for _ in range(2):
        f_pred = f(x_next, y_corr)
        y_corr = y_curr + (h / 2.0) * (f_pred + f_curr)

    return y_corr



#  Ч.2. ПУНКТ 6: Метод Рунге-Кутта 4-го порядку (фіксований крок)

def runge_kutta_4(f, x0, y0, x_end, h):
    #Класичний метод Рунге-Кутта 4-го порядку
    steps = int((x_end - x0) / h)
    x_pts = [x0]
    y_pts = [y0]

    for i in range(steps):
        xi = x_pts[-1]
        yi = y_pts[-1]
        y_next = rk4_step(f, xi, yi, h)
        x_pts.append(xi + h)
        y_pts.append(y_next)

    return np.array(x_pts), np.array(y_pts)


# Ч.1. ПУНКТ 2: Метод прогнозу та корекції Адамса 2-го порядку

def adams_predictor_corrector_2(f, x0, y0, x_end, h):

    #Двокроковий метод Адамса (прогноз-корекція)

    steps = int((x_end - x0) / h)
    x_pts = [x0]
    y_pts = [y0]

    # Для старту двокрокового методу Адамса потрібна ще одна точка y1.

    y1 = rk4_step(f, x0, y0, h)
    x_pts.append(x0 + h)
    y_pts.append(y1)

    # Основний цикл методу Адамса 2-го порядку
    for i in range(1, steps):
        y_next = adams_step(f, x_pts[i - 1], y_pts[i - 1], x_pts[i], y_pts[i], h)
        x_pts.append(x_pts[i] + h)
        y_pts.append(y_next)

    return np.array(x_pts), np.array(y_pts)



#  Ч.1. ПУНКТ 5: Автоматичний вибір кроку для Адамса

def adams_adaptive(f, x0, y0, x_end, eps, h_init=0.1):

   # Метод Адамса 2-го порядку з автоматичним вибором кроку
    #Використовує оцінку похибки через порівняння з кроком h/2

    x_pts = [x0]
    y_pts = [y0]
    h_hist = []

    current_h = h_init
    x_curr = x0
    y_curr = y0

    # Для старту потрібна друга точка
    y1 = rk4_step(f, x_curr, y_curr, current_h)

    x_prev = x_curr
    y_prev = y_curr
    x_curr = x_curr + current_h
    y_curr = y1

    while x_curr < x_end:
        if x_curr + current_h > x_end:
            current_h = x_end - x_curr

        # Обчислення наступного значення з кроком current_h
        y_next_h = adams_step(f, x_prev, y_prev, x_curr, y_curr, current_h)

        # Обчислення з кроком current_h/2 (потрібно зробити два кроки)
        # Перший крок
        y_mid = adams_step(f, x_prev, y_prev, x_curr, y_curr, current_h / 2)
        x_mid = x_curr + current_h / 2
        # Другий крок
        y_next_half = adams_step(f, x_curr, y_curr, x_mid, y_mid, current_h / 2)

        # Оцінка похибки (для Адамса 2-го порядку)
        error_est = abs(y_next_half - y_next_h)

        if error_est < eps:
            # Крок прийнято
            x_prev = x_curr
            y_prev = y_curr
            x_curr = x_curr + current_h
            y_curr = y_next_half  # Беремо більш точне значення з кроком h/2

            # Зберігаємо результати
            x_pts.append(x_curr)
            y_pts.append(y_curr)
            h_hist.append(current_h)

            # Якщо похибка дуже мала, збільшуємо крок
            if error_est < eps / 10:
                current_h = min(current_h * 2, x_end - x_curr)

                if current_h < 1e-8:
                    current_h = 1e-8
        else:
            # Похибка завелика - зменшуємо крок
            current_h = current_h / 2

    return np.array(x_pts), np.array(y_pts), np.array(h_hist)



#  Ч.2. ПУНКТ 9: Автоматичний вибір кроку для Рунге-Кутта

def runge_kutta_4_adaptive(f, x0, y0, x_end, eps, h_init=0.1):

    #Метод Рунге-Кутта 4-го порядку

    x_pts = [x0]
    y_pts = [y0]
    h_hist = []

    current_h = h_init
    x_curr = x0
    y_curr = y0

    while x_curr < x_end:
        if x_curr + current_h > x_end:
            current_h = x_end - x_curr


        # Один крок з кроком h
        y_h = rk4_step(f, x_curr, y_curr, current_h)

        # Два кроки з кроком h/2
        y_half1 = rk4_step(f, x_curr, y_curr, current_h / 2)
        y_half2 = rk4_step(f, x_curr + current_h / 2, y_half1, current_h / 2)

        # Оцінка похибки за Рунге для 4-го порядку
        error_est = (16.0 / 15.0) * abs(y_h - y_half2)

        if error_est < eps:
            # Крок прийнято
            x_curr += current_h
            y_curr = y_half2  # Більш точне значення

            x_pts.append(x_curr)
            y_pts.append(y_curr)
            h_hist.append(current_h)

            # Якщо похибка дуже мала, збільшуємо крок
            if error_est < eps / 32.0:
                current_h = min(current_h * 2, x_end - x_curr)
        else:
            # Похибка завелика - зменшуємо крок
            current_h = current_h / 2.0

    return np.array(x_pts), np.array(y_pts), np.array(h_hist)


# ОБЧИСЛЕННЯ ДЛЯ ПОХИБОК МЕТОДІВ З ФІКСОВАНИМ КРОКОМ
x_rk, y_rk = runge_kutta_4(f, x0, y0, x_end, h)
x_ad, y_ad = adams_predictor_corrector_2(f, x0, y0, x_end, h)

y_exact_rk = exact_sol(x_rk)
y_exact_ad = exact_sol(x_ad)

# Локальні похибки
err_rk_exact = np.abs(y_exact_rk - y_rk)

err_ad_exact = np.abs(y_exact_ad - y_ad)


# Ч.1. ПУНКТ 4: Оцінка похибки Адамса через теоретичну формулу

err_ad_theoretical = []
for x in x_ad:
    y_triple_prime = 0.5 * np.exp(x)  # Третя похідна точного розв'язку
    r_kop = np.abs(-(h ** 3 / 12.0) * y_triple_prime)
    err_ad_theoretical.append(r_kop)
err_ad_theoretical = np.array(err_ad_theoretical)


#  Ч.2. ПУНКТ 8: Оцінка похибки Рунге-Кутта за методом Рунге

err_rk_runge = [0.0]
for i in range(1, len(x_rk)):
    xi_prev = x_rk[i - 1]
    yi_prev = y_rk[i - 1]

    y_h = y_rk[i]
    y_half1 = rk4_step(f, xi_prev, yi_prev, h / 2)
    y_half2 = rk4_step(f, xi_prev + h / 2, y_half1, h / 2)

    runge_err = (16.0 / 15.0) * np.abs(y_h - y_half2)
    err_rk_runge.append(runge_err)
err_rk_runge = np.array(err_rk_runge)


#  Адаптивний метод Адамса
x_ad_adapt, y_ad_adapt, h_ad_adapt = adams_adaptive(f, x0, y0, x_end, eps)
#  Адаптивний метод Рунге-Кутта
x_rk_adapt, y_rk_adapt, h_rk_adapt = runge_kutta_4_adaptive(f, x0, y0, x_end, eps)


#  ГРАФІКИ
# Графік 1: Локальні похибки
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Лівий графік — Метод Адамса (Пункти 3,4)
ax1.semilogy(x_ad, err_ad_exact, 'b-o', label="Фактична локальна похибка")
ax1.semilogy(x_ad, err_ad_theoretical, 'r--', label="Теоретична оцінка (R_kop)")
ax1.set_title("Пункти 3,4: Похибка методу Адамса 2-го порядку")
ax1.set_xlabel("x")
ax1.set_ylabel("Похибка (логарифмічна шкала)")
ax1.grid(True)
ax1.legend()

# Правий графік — Метод Рунге-Кутта
ax2.semilogy(x_rk, err_rk_exact, 'g-s', label="Фактична локальна похибка")
ax2.semilogy(x_rk, err_rk_runge, 'm--', label="Оцінка за методом Рунге")
ax2.set_title("Пункти 7,8: Похибка методу Рунге-Кутта 4-го порядку")
ax2.set_xlabel("x")
ax2.set_ylabel("Похибка (логарифмічна шкала)")
ax2.grid(True)
ax2.legend()
plt.show()

# Графік 2: Автоматичний вибір кроку для Адамса
plt.figure(figsize=(10, 4))
plt.step(x_ad_adapt[:-1], h_ad_adapt, where='post', color='blue', linewidth=2, label="Адаптивний крок Адамса")
plt.title("Пункт 5: Залежність величини кроку h від x (метод Адамса)")
plt.xlabel("x")
plt.ylabel("Крок h")
plt.grid(True)
plt.legend()
plt.show()

# Графік 3: Автоматичний вибір кроку для Рунге-Кутта
plt.figure(figsize=(10, 4))
plt.step(x_rk_adapt[:-1], h_rk_adapt, where='post', color='orange', linewidth=2, label="Адаптивний крок Рунге-Кутта")
plt.title("Пункт 9: Залежність величини кроку h від x (метод Рунге-Кутта 4)")
plt.xlabel("x")
plt.ylabel("Крок h")
plt.grid(True)
plt.legend()
plt.show()

# Порівняння адаптивних методів
plt.figure(figsize=(10, 4))
plt.step(x_ad_adapt[:-1], h_ad_adapt, where='post', color='blue', linewidth=2, label="Адамс адаптивний")
plt.step(x_rk_adapt[:-1], h_rk_adapt, where='post', color='orange', linewidth=2, label="Рунге-Кутта адаптивний")
plt.title("Порівняння адаптивних кроків")
plt.xlabel("x")
plt.ylabel("Крок h")
plt.grid(True)
plt.legend()
plt.show()

# Виведення результатів у консоль
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТИ")
print("=" * 60)
print(f"Рівняння: dy/dx = y - x")
print(f"Початкова умова: y({x0}) = {y0}")
print(f"Відрізок: x ∈ [{x0}, {x_end}]")
print(f"Точний розв'язок: y = 0.5*e^x + x + 1")
print("-" * 60)
print(f"Фіксований крок h = {h}")
print(f"   Кількість кроків Рунге-Кутта: {len(x_rk) - 1}")
print(f"   Кількість кроків Адамса: {len(x_ad) - 1}")
print(f"   Макс. похибка Рунге-Кутта: {np.max(err_rk_exact):.2e}")
print(f"   Макс. похибка Адамса: {np.max(err_ad_exact):.2e}")
print("-" * 60)
print(f"Адаптивний метод (точність eps={eps})")
print(f"   Кількість кроків Адамса (адаптивний): {len(x_ad_adapt) - 1}")
print(f"   Кількість кроків Рунге-Кутта (адаптивний): {len(x_rk_adapt) - 1}")
print(f"   Мінімальний крок Адамса: {np.min(h_ad_adapt):.2e}")
print(f"   Мінімальний крок Рунге-Кутта: {np.min(h_rk_adapt):.2e}")
print(f"   Максимальний крок Адамса: {np.max(h_ad_adapt):.2e}")
print(f"   Максимальний крок Рунге-Кутта: {np.max(h_rk_adapt):.2e}")
print("=" * 60)
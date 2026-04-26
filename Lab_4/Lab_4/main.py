import numpy as np
import matplotlib.pyplot as plt

# Задання функції вологості ґрунту
def M(t):
    "Функція вологості ґрунту"
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def M_prime_exact(t):
    "Точна аналітична похідна функції вологості"
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)


# Завдання 1: Аналітичне розв'язання

print("=" * 60)
print("ЗАВДАННЯ 1: Аналітичне розв'язання")
print("=" * 60)

# Точка, в якій обчислюємо похідну
t0 = 1.0

# Обчислення точної похідної
exact_derivative = M_prime_exact(t0)

print(f"Точка обчислення: t0 = {t0}")
print(f"Точне значення похідної M'({t0}) = {exact_derivative:.6f}")
print()


# Завдання 2: Дослідження залежності похибки від кроку h

print("=" * 60)
print("ЗАВДАННЯ 2: Дослідження залежності похибки від кроку h")
print("=" * 60)

def central_difference(t, h, func):
    "Центральна різницева формула для першої похідної"
    return (func(t + h) - func(t - h)) / (2 * h)

# Створюємо масив кроків від 10^-20 до 10^3
h_values = np.logspace(-20, 3, 100)
errors = []

for h in h_values:
    approx = central_difference(t0, h, M)
    error = abs(approx - exact_derivative)
    errors.append(error)

# Знаходимо оптимальний крок (той, що дає мінімальну похибку)
min_error_index = np.argmin(errors)
optimal_h = h_values[min_error_index]
min_error = errors[min_error_index]

print(f"Оптимальний крок h0 = {optimal_h:.2e}")
print(f"Досягнута точність R0 = {min_error:.6e}")
print()

# Побудова графіка залежності похибки від кроку
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors, 'b-', linewidth=2)
plt.loglog(optimal_h, min_error, 'ro', markersize=8, label=f'Оптимум: h={optimal_h:.2e}')
plt.xlabel('Крок h')
plt.ylabel('Похибка |M\'_числ - M\'_точн|')
plt.title('Залежність похибки чисельного диференціювання від кроку h')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# Завдання 3: Прийняти h = 10^-3

print("=" * 60)
print("ЗАВДАННЯ 3: Приймаємо h = 10^-3")
print("=" * 60)

h = 1e-3  # фіксований крок
print(f"Вибраний крок h = {h}")
print()


# Завдання 4: Обчислення похідної з кроками h та 2h

print("=" * 60)
print("ЗАВДАННЯ 4: Обчислення похідної з кроками h та 2h")
print("=" * 60)

# Обчислення з кроком h
approx_h = central_difference(t0, h, M)

# Обчислення з кроком 2h
approx_2h = central_difference(t0, 2 * h, M)

print(f"y0'(h)  = {approx_h:.8f}")
print(f"y0'(2h) = {approx_2h:.8f}")
print()


# Завдання 5: Обчислення похибки при кроці h

print("=" * 60)
print("ЗАВДАННЯ 5: Похибка при кроці h")
print("=" * 60)

error_h = abs(approx_h - exact_derivative)
print(f"Похибка R1 = |y0'(h) - M'(t0)| = {error_h:.8e}")
print(f"Точне значення M'({t0}) = {exact_derivative:.8f}")
print(f"Наближене значення y0'(h) = {approx_h:.8f}")
print()


# Завдання 6: Метод Рунге-Ромберга

print("=" * 60)
print("ЗАВДАННЯ 6: Метод Рунге-Ромберга")
print("=" * 60)

# Метод Рунге-Ромберга: уточнене значення
# Формула: y_R' = y0'(h) + (y0'(h) - y0'(2h)) / (2^2 - 1)
# Оскільки p=2 для центральної різниці, то q=2, q^p - 1 = 4 - 1 = 3
runge_romberg = approx_h + (approx_h - approx_2h) / 3

error_runge = abs(runge_romberg - exact_derivative)

print(f"Уточнене значення (Рунге-Ромберг): y_R' = {runge_romberg:.8f}")
print(f"Похибка R2 = |y_R' - M'(t0)| = {error_runge:.8e}")
print()
print("Аналіз зміни похибки:")
print(f"  Похибка до уточнення: {error_h:.6e}")
print(f"  Похибка після уточнення: {error_runge:.6e}")
print(f"  Похибка зменшилась у {error_h/error_runge:.1f} разів")
print()

# ============================================================
# Завдання 7: Метод Ейткена (з кроками h, 2h, 4h)
# ============================================================
print("=" * 60)
print("ЗАВДАННЯ 7: Метод Ейткена")
print("=" * 60)

# Обчислення похідної з кроком 4h
approx_4h = central_difference(t0, 4 * h, M)

print(f"y0'(h)  = {approx_h:.8f}")
print(f"y0'(2h) = {approx_2h:.8f}")
print(f"y0'(4h) = {approx_4h:.8f}")
print()

# Метод Ейткена для уточнення значення похідної
# Формула: y_E' = ( (y'(2h))^2 - y'(4h)*y'(h) ) / ( 2*y'(2h) - (y'(4h) + y'(h)) )
numerator = approx_2h**2 - approx_4h * approx_h
denominator = 2 * approx_2h - (approx_4h + approx_h)
aitken = numerator / denominator

# Оцінка порядку точності
# Формула: p = ln| (y'(4h) - y'(2h)) / (y'(2h) - y'(h)) | / ln(2)
p_estimate = np.log(abs((approx_4h - approx_2h) / (approx_2h - approx_h))) / np.log(2)

error_aitken = abs(aitken - exact_derivative)

print(f"Уточнене значення (Ейткен): y_E' = {aitken:.8f}")
print(f"Похибка R3 = |y_E' - M'(t0)| = {error_aitken:.8e}")
print(f"Оцінка порядку точності p = {p_estimate:.4f}")
print()
print("Аналіз зміни похибки:")
print(f"  Похибка при кроці h: {error_h:.6e}")
print(f"  Похибка після Рунге-Ромберга: {error_runge:.6e}")
print(f"  Похибка після Ейткена: {error_aitken:.6e}")

# ============================================================
#  Порівняння результатів
# ============================================================
print()
print("=" * 60)
print("ПОРІВНЯЛЬНИЙ АНАЛІЗ")
print("=" * 60)

print(f"{'Метод':<20} {'Значення похідної':<20} {'Похибка':<15}")
print("-" * 55)
print(f"{'Точне значення':<20} {exact_derivative:<20.8f} {'---':<15}")
print(f"{'Центр. різниця (h)':<20} {approx_h:<20.8f} {error_h:<15.2e}")
print(f"{'Рунге-Ромберг':<20} {runge_romberg:<20.8f} {error_runge:<15.2e}")
print(f"{'Ейткен':<20} {aitken:<20.8f} {error_aitken:<15.2e}")

# Висновок про оптимальні режими поливу
print()
print("=" * 60)
print("ВИСНОВОК ПРО ОПТИМАЛЬНІ РЕЖИМИ ПОЛИВУ")
print("=" * 60)
print(f"Швидкість зміни вологості ґрунту в момент t = {t0} становить M'({t0}) ≈ {exact_derivative:.4f}")
print(f"Оскільки значення від'ємне, вологість зменшується (ґрунт висихає).")
print()
print("Рекомендації щодо поливу:")
if exact_derivative < -2:
    print("  - Висока швидкість висихання. Полив необхідно ввімкнути найближчим часом.")
elif exact_derivative < -1:
    print("  - Середня швидкість висихання. Полив рекомендується протягом найближчих годин.")
elif exact_derivative < 0:
    print("  - Низька швидкість висихання. Полив можна відкласти.")
else:
    print("  - Вологість зростає. Полив не потрібен.")
print()
print(f"Використання методів підвищення точності (Рунге-Ромберга та Ейткена)")
print(f"дозволило зменшити похибку обчислення швидкості висихання з")
print(f"{error_h:.2e} до {error_aitken:.2e}, що забезпечує більш точне")
print("визначення моменту ввімкнення системи поливу.")
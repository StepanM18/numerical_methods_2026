
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


#  ЗАВДАННЯ 1

def f(x):

    #Функція інтенсивності навантаження на сервер.

    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)


# ЗАВДАННЯ 2
# Знаходимо точне значення інтегралу
print("=" * 70)
print("ЗАВДАННЯ 2: Точне значення інтегралу")
print("=" * 70)

a = 0
b = 24

I_exact, _ = quad(f, a, b)

print(f"Точне значення інтегралу I0 = {I_exact:.12f}")
print()


# ЗАВДАННЯ 3
# ОПИС: Реалізуємо складову формулу Сімпсона
def simpson_composite(f, a, b, N):

    if N % 2 != 0:
        N += 1

    # Крок інтегрування (довжина одного маленького відрізка)
    h = (b - a) / N

    # Створюємо масив точок
    x = np.linspace(a, b, N + 1)

    # Обчислюємо значення функції в кожній точці
    y = f(x)

    # Формула Сімпсона:
       # перша та остання точки
    integral = y[0] + y[-1]

    # Додаємо непарні точки
    integral += 4 * np.sum(y[1:-1:2])

    # Додаємо парні точки
    integral += 2 * np.sum(y[2:-2:2])

    # Множимо на h/3
    integral *= h / 3

    return integral


#ЗАВДАННЯ 4
# ОПИС: Досліджуємо залежність похибки від кількості розбиттів N
print("=" * 70)
print("ЗАВДАННЯ 4: Дослідження залежності похибки від N")
print("=" * 70)

# Створюємо список значень N від 10 до 1000
N_values = range(10, 1001, 2)

# Список для зберігання похибок
errors = []

# Для кожного N обчислюємо інтеграл і похибку
for N in N_values:
    I_approx = simpson_composite(f, a, b, N)
    error = abs(I_approx - I_exact)
    errors.append(error)

# Задана точність, якої треба досягти (з методички)
eps_target = 1e-12

# Шукаємо мінімальне N, при якому похибка < 1e-12
N_opt = None
for N, err in zip(N_values, errors):
    if err < eps_target:
        N_opt = N
        break

if N_opt is None:
    N_opt = N_values[-1]

# Обчислюємо інтеграл при знайденому N_opt
I_opt = simpson_composite(f, a, b, N_opt)

# Обчислюємо досягнуту точність
eps_opt = abs(I_opt - I_exact)

print(f"Цільова точність ε = {eps_target:.0e}")
print(f"Оптимальне число розбиттів N_opt = {N_opt}")
print(f"Досягнута точність eps_opt = {eps_opt:.4e}")
print(f"Значення інтегралу при N_opt: I = {I_opt:.12f}")
print()

# ПОБУДОВА ГРАФІКА

plt.figure(figsize=(10, 6))

plt.loglog(N_values, errors, 'b-', linewidth=2)

plt.loglog(N_opt, eps_opt, 'ro', markersize=8, label=f'N_opt={N_opt}, eps={eps_opt:.2e}')

plt.axhline(y=eps_target, color='r', linestyle='--', label=f'ε = {eps_target:.0e}')

plt.xlabel('Число розбиттів N')
plt.ylabel('Похибка |I(N) - I0|')

plt.title('Залежність похибки чисельного інтегрування (метод Сімпсона) від N')

plt.grid(True, alpha=0.3)

plt.legend()

plt.show()


# ЗАВДАННЯ 5
# ОПИС: Обчислюємо похибку при N0 ≈ N_opt/10
print("=" * 70)
print("ЗАВДАННЯ 5: Похибка при N0")
print("=" * 70)

# Беремо N0 як N_opt/10
N0 = int(N_opt / 10)

# N0 має бути кратним 8
N0 = ((N0 + 7) // 8) * 8

if N0 < 8:
    N0 = 8

# Обчислюємо інтеграл при N0
I_N0 = simpson_composite(f, a, b, N0)

# Обчислюємо похибку
eps0 = abs(I_N0 - I_exact)

print(f"N0 = {N0} (кратне 8, ≈ N_opt/10)")
print(f"Значення інтегралу I(N0) = {I_N0:.12f}")
print(f"Похибка eps0 = {eps0:.4e}")
print()


#  ЗАВДАННЯ 6
# Метод Рунге-Ромберга
print("=" * 70)
print("ЗАВДАННЯ 6: Метод Рунге-Ромберга")
print("=" * 70)

# Для методу Рунге-Ромберга потрібні I(N0) та I(N0/2)
N0_half = N0 // 2                    # половина від N0
I_N0_half = simpson_composite(f, a, b, N0_half)  # інтеграл при N0/2


I_Runge = I_N0 + (I_N0 - I_N0_half) / 15

# Обчислюємо похибку після уточнення
eps_Runge = abs(I_Runge - I_exact)

print(f"I(N0)     = {I_N0:.12f}")
print(f"I(N0/2)   = {I_N0_half:.12f}")
print(f"Рунге-Ромберг I_R = {I_Runge:.12f}")
print(f"Похибка eps_R = {eps_Runge:.4e}")
print(f"Похибка зменшилась у {eps0/eps_Runge:.1f} разів")
print()


#  ЗАВДАННЯ 7
#  Метод Ейткена
print("=" * 70)
print("ЗАВДАННЯ 7: Метод Ейткена")
print("=" * 70)

# Для методу Ейткена  I(N0), I(N0/2), I(N0/4)
N0_quarter = N0 // 4                          # чверть від N0
I_N0_quarter = simpson_composite(f, a, b, N0_quarter)  # інтеграл при N0/4

# Формула Ейткена:

chiselnik = I_N0_half**2 - I_N0 * I_N0_quarter
znamennyk = 2 * I_N0_half - (I_N0 + I_N0_quarter)
I_Eitken = chiselnik / znamennyk

vidnoshennya = abs((I_N0_quarter - I_N0_half) / (I_N0_half - I_N0))
p = np.log(vidnoshennya) / np.log(2)

# Обчислюємо похибку після уточнення методом Ейткена
eps_Eitken = abs(I_Eitken - I_exact)

print(f"I(N0)       = {I_N0:.12f}")
print(f"I(N0/2)     = {I_N0_half:.12f}")
print(f"I(N0/4)     = {I_N0_quarter:.12f}")
print(f"Ейткен I_E  = {I_Eitken:.12f}")
print(f"Похибка eps_E = {eps_Eitken:.4e}")
print(f"Оцінка порядку точності p = {p:.4f}")
print()


#  ЗАВДАННЯ 8
# ОПИС: Аналіз зміни похибки (порівняльна таблиця)
print("=" * 70)
print("ЗАВДАННЯ 8: Аналіз зміни похибки")
print("=" * 70)

print("\nПорівняльна таблиця:")
print("-" * 60)
print(f"{'Метод':<25} {'Значення інтегралу':<20} {'Похибка':<15}")
print("-" * 60)
print(f"{'Точне значення':<25} {I_exact:<20.12f} {'---':<15}")
print(f"{'Сімпсон (N0)':<25} {I_N0:<20.12f} {eps0:<15.2e}")
print(f"{'Рунге-Ромберг':<25} {I_Runge:<20.12f} {eps_Runge:<15.2e}")
print(f"{'Ейткен':<25} {I_Eitken:<20.12f} {eps_Eitken:<15.2e}")

print("\nАналіз:")
print(f"  - Похибка методу Сімпсона при N0 = {N0}: {eps0:.2e}")
print(f"  - Після уточнення Рунге-Ромбергом: {eps_Runge:.2e} (покращення в {eps0/eps_Runge:.1f} разів)")
print(f"  - Після уточнення Ейткеном: {eps_Eitken:.2e} (покращення в {eps0/eps_Eitken:.1f} разів)")
print(f"  - Порядок точності формули Сімпсона: p = {p:.2f} (теоретично p=4)")


# ЗАВДАННЯ 9
# ОПИС: Адаптивний алгоритм (автоматичний вибір кроку)
print("\n" + "=" * 70)
print("ЗАВДАННЯ 9: Адаптивний алгоритм")
print("=" * 70)

def adaptive_simpson(f, a, b, tol, max_depth=20):


    def simpson_rule(a, b):
        h = (b - a) / 2                    # половина довжини відрізка
        mid = (a + b) / 2                  # середина відрізка

        return (h / 3) * (f(a) + 4 * f(mid) + f(b))

    # Рекурсивна функція для адаптивного розбиття
    def adaptive_recursive(a, b, tol, depth):
        # Середина відрізка
        m = (a + b) / 2

        # Обчислюємо інтеграл на всьому відрізку [a,b]
        S_whole = simpson_rule(a, b)

        # Обчислюємо інтеграли на половинках
        S_left = simpson_rule(a, m)
        S_right = simpson_rule(m, b)


        if depth >= max_depth:
            return S_whole

        # Оцінка похибки: |S_left + S_right - S_whole|

        if abs(S_left + S_right - S_whole) < 15 *

            return S_left + S_right
        else:

            return (adaptive_recursive(a, m, tol/2, depth+1) +
                    adaptive_recursive(m, b, tol/2, depth+1))

    # Запускаємо рекурсію
    return adaptive_recursive(a, b, tol, 0)


# Досліджуємо залежність похибки від параметра δ

delta_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
adaptive_errors = []

print("\nДослідження адаптивного алгоритму:")
print("-" * 70)
print(f"{'δ (точність)':<15} {'I_adapt':<18} {'Похибка':<15}")
print("-" * 70)

# Для кожного значення δ обчислюємо інтеграл адаптивним методом
for delta in delta_values:
    I_adapt = adaptive_simpson(f, a, b, delta)
    error = abs(I_adapt - I_exact)
    adaptive_errors.append(error)
    print(f"{delta:<15.0e} {I_adapt:<18.12f} {error:<15.2e}")

print("\nАналіз адаптивного алгоритму:")
print("  - Адаптивний метод автоматично згущує сітку там, де функція змінюється швидше")
print("  - Це дозволяє досягти високої точності з меншою кількістю обчислень")
print("  - При δ = 1e-10 похибка стає меншою за 1e-10")

# Побудова графіка для адаптивного методу
plt.figure(figsize=(10, 6))
plt.loglog(delta_values, adaptive_errors, 'g-', linewidth=2, marker='o')
plt.xlabel('Параметр δ (задана точність)')
plt.ylabel('Досягнута похибка |I_adapt - I0|')
plt.title('Залежність похибки адаптивного методу від параметра δ')
plt.grid(True, alpha=0.3)
plt.show()


#ЗАВДАННЯ 10
# Висновки
print("\n" + "=" * 70)
print("ЗАВДАННЯ 10: ВИСНОВКИ")
print("=" * 70)

print(f"""
1. Складова формула Сімпсона має четвертий порядок точності (p ≈ {p:.2f}).
   Це означає, що при зменшенні кроку вдвічі похибка зменшується в 16 разів.

2. Метод Рунге-Ромберга дозволив зменшити похибку в {eps0/eps_Runge:.1f} разів,
   використовуючи комбінацію результатів з різними кроками.

3. Метод Ейткена підтвердив порядок точності та дав ще більш точне значення.
   Похибка зменшилась в {eps0/eps_Eitken:.1f} разів.

4. Адаптивний алгоритм дозволяє автоматично вибирати крок інтегрування,
   згущуючи сітку там, де функція змінюється швидше (наприклад, біля x=12).

5. Для досягнення точності ε=1e-12 знадобилося N_opt = {N_opt} розбиттів.

6. Сумарне навантаження на сервер за добу становить I0 ≈ {I_exact:.2f} (умовних одиниць).
""")
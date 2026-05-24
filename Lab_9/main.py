
# ЛАБОРАТОРНА РОБОТА №9
# Метод Хука-Дживса багатовимірної оптимізації


import math
import matplotlib.pyplot as plt



#  1


def f1(x, y):

    return x ** 2 + y ** 2 - 4


def f2(x, y):

    return x - y - 1



# ПОБУДОВА ЦІЛЬОВОЇ ФУНКЦІЇ



def phi(point):

    x = point[0]
    y = point[1]

    return f1(x, y) ** 2 + f2(x, y) ** 2



# ПУНКТ 1
# "Побудувати графіки рівнянь"

def draw_graphs():

    #Побудова графіків системи рівнянь


    x_values = []
    y1_upper = []
    y1_lower = []
    y2_values = []


    points = []
    x = -2.0
    step = 0.01

    while x <= 2.0:
        points.append(x)
        x += step

    if abs(points[-1] - 2.0) > 0.001:
        points.append(2.0)

    for x in points:
        # Коло: y = ±√(4 - x²)
        y_upper = math.sqrt(max(0, 4 - x ** 2))
        y_lower = -y_upper

        x_values.append(x)
        y1_upper.append(y_upper)
        y1_lower.append(y_lower)
        y2_values.append(x - 1)

    plt.figure(figsize=(8, 6))

    plt.plot(x_values, y1_upper, 'b-', linewidth=2, label='x² + y² = 4')
    plt.plot(x_values, y1_lower, 'b-', linewidth=2)
    plt.plot(x_values, y2_values, 'r-', linewidth=2, label='y = x - 1')


    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Графіки рівнянь системи")
    plt.grid(True, alpha=0.3)


    plt.legend(loc='upper right')

    plt.xlim(-2.2, 2.2)
    plt.ylim(-2.2, 2.2)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linewidth=0.5)

    plt.show()


# ПУНКТ 2
# "Написати програму знаходження мінімуму
# цільової функції методом Хука-Дживса"



def exploratory_search(base_point, delta):

    #ДОСЛІДЖУЮЧИЙ ПОШУК



    # Копіюємо базисну точку
    new_point = base_point.copy()


    # Цикл по всіх координатах

    for i in range(len(new_point)):

        # Обчислюємо значення функції
        current_value = phi(new_point)


        # Перевірка руху ВПЕРЕД


        temp = new_point.copy()

        temp[i] = temp[i] + delta[i]

        if phi(temp) < current_value:
            new_point = temp

        else:


            # Якщо вперед не покращилось,
            # пробуємо рух НАЗАД


            temp = new_point.copy()

            temp[i] = temp[i] - delta[i]

            if phi(temp) < current_value:
                new_point = temp

    return new_point


# ОСНОВНА ФУНКЦІЯ МЕТОДУ ХУКА-ДЖИВСА


def hooke_jeeves(
        start_point,
        delta,
        alpha,
        epsilon,
        max_iterations=1000
):



    # Початкова базисна точка


    base_point = start_point.copy()

    # Список точок траєкторії
    trajectory = []

    trajectory.append(base_point.copy())

    iteration = 0


    # ГОЛОВНИЙ ЦИКЛ АЛГОРИТМУ


    while max(delta) > epsilon and iteration < max_iterations:

        iteration += 1


        # ДОСЛІДЖУЮЧИЙ ПОШУК


        new_point = exploratory_search(base_point, delta)


        # Якщо знайдено кращу точку


        if phi(new_point) < phi(base_point):


            # ПОШУК ПО ЗРАЗКУ
            #
            # x_new = x2 + (x2 - x1)


            pattern_point = []

            for i in range(len(base_point)):
                value = new_point[i] + (new_point[i] - base_point[i])
                pattern_point.append(value)


            # Дослідження нової точки


            pattern_point = exploratory_search(pattern_point, delta)


            # Оновлення базисної точки

            if phi(pattern_point) < phi(new_point):
                base_point = pattern_point
            else:
                base_point = new_point

        else:


            # Якщо покращення немає,
            # зменшуємо крок


            for i in range(len(delta)):
                delta[i] = delta[i] / alpha

        trajectory.append(base_point.copy())

    return base_point, trajectory



# ПУНКТ 3
# "Протестувати програму"


print("================================================")
print("ЛАБОРАТОРНА РОБОТА №9")
print("Метод Хука-Дживса")
print("================================================")


# ПУНКТ 4


# Початкова точка
start_point = [1.0, 1.0]

# Початкові кроки
delta = [0.5, 0.5]

# Коефіцієнт зменшення кроку
alpha = 2

# Точність
epsilon = 0.0001


# ВИКЛИК МЕТОДУ ХУКА-ДЖИВСА


solution, trajectory = hooke_jeeves(
    start_point,
    delta,
    alpha,
    epsilon
)


# ВИВЕДЕННЯ РЕЗУЛЬТАТІВ


print("\nРозв'язок системи:")

print("x =", round(solution[0], 6))
print("y =", round(solution[1], 6))

print("\nЗначення цільової функції:")
print("Φ =", phi(solution))


# ПУНКТ 5
# "Вивести в файл координати точок траєкторії"


file = open("trajectory.txt", "w", encoding="utf-8")

file.write("Траєкторія спуску:\n\n")

step = 0

for point in trajectory:

    file.write(
        f"Крок {step}: "
        f"x = {point[0]:.6f}, "
        f"y = {point[1]:.6f}, "
        f"Φ = {phi(point):.10f}\n"
    )

    step += 1

file.close()


# Виведення кількості кроків

print("\nКількість кроків:")
print(len(trajectory))

print("\nКоординати траєкторії записані у файл trajectory.txt")


# ПОБУДОВА ГРАФІКІВ


draw_graphs()
import requests
import numpy as np
import matplotlib.pyplot as plt

# 1. ЗАПИТ ДО OPEN-ELEVATION API
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
data = response.json()
results = data["results"]

# 2. ТАБУЛЯЦІЯ ВУЗЛІВ ТА ЗАПИС У ФАЙЛ
n = len(results)
print("Кількість вузлів:", n)

# Вивід у консоль
print("\nТабуляція вузлів:")
print(" № |  Latitude | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")

# Функція гаверсинуса для відстані між GPS координатами
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# Обчислення кумулятивної відстані
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])
distances = [0]
for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)
distances = np.array(distances)

# Запис у файл
with open("tabulation.txt", "w") as f:
    f.write(" № |  Latitude   |  Longitude  | Elevation (m) | Distance (m)\n")
    for i in range(n):
        f.write(f"{i:2d} | {coords[i][0]:.6f} | {coords[i][1]:.6f} | "
                f"{elevations[i]:.2f} | {distances[i]:.2f}\n")

print("\nТабуляцію збережено у файл 'tabulation.txt'.")

# 3. РЕАЛІЗАЦІЯ КУБІЧНОГО СПЛАЙНА
def cubic_spline_natural(x, y): #x-масив відстаней,у-висот

    n = len(x)
    h = np.diff(x)

    # Ініціалізація тридіагональної матриці (піддіагональ A, головна B, наддіагональ C)
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n) #права частина

    # Крайові умови натурального сплайна (c1 = 0, cn = 0)
    B[0] = 1
    B[-1] = 1

    # Заповнення внутрішніх рівнянь
    for i in range(1, n-1):
        A[i] = h[i-1]
        B[i] = 2 * (h[i-1] + h[i])
        C[i] = h[i]
        D[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

    # Пряма прогонка (метод Томаса)
    for i in range(1, n):
        m = A[i] / B[i-1]
        B[i] -= m * C[i-1]
        D[i] -= m * D[i-1]

    # Зворотна прогонка
    c = np.zeros(n)
    c[-1] = D[-1] / B[-1]
    for i in range(n-2, -1, -1):
        c[i] = (D[i] - C[i] * c[i+1]) / B[i]

    # Обчислення коефіцієнтів a, b, d
    a = y[:-1]
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    for i in range(n-1):
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (2*c[i] + c[i+1]) / 6
        d[i] = (c[i+1] - c[i]) / (6 * h[i])


    c_interval = c[:-1] / 2.0

    return a, b, c_interval, d, x

# Функція для обчислення значення сплайна в довільній точці
def spline_eval(xi, a, b, c, d, x_nodes):
    for i in range(len(x_nodes)-1):
        if x_nodes[i] <= xi <= x_nodes[i+1]:
            dx = xi - x_nodes[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return None

# Побудова сплайна за всіма 21 вузлами
a_all, b_all, c_all, d_all, x_all = cubic_spline_natural(distances, elevations)

# Вивід коефіцієнтів
print("\n" + "="*60)
print("Коефіцієнти кубічних сплайнів (для кожного інтервалу):")
print("="*60)
for i in range(n-1):
    print(f"Інтервал [{distances[i]:.2f}, {distances[i+1]:.2f}]:")
    print(f"  a[{i}] = {a_all[i]:.4f}")
    print(f"  b[{i}] = {b_all[i]:.4f}")
    print(f"  c[{i}] = {c_all[i]:.6f}")
    print(f"  d[{i}] = {d_all[i]:.8f}")
    print()

# 4. ПОРІВНЯННЯ ДЛЯ РІЗНОЇ КІЛЬКОСТІ ВУЗЛІВ

xx = np.linspace(distances[0], distances[-1], 1000)
yy_all = np.array([spline_eval(xi, a_all, b_all, c_all, d_all, x_all) for xi in xx])

def test_nodes(k):
    # Вибираємо k рівномірно розподілених вузлів
    indices = np.linspace(0, n-1, k, dtype=int)
    x_k = distances[indices]
    y_k = elevations[indices]
    a, b, c, d, x_nodes = cubic_spline_natural(x_k, y_k)
    yy = np.array([spline_eval(xi, a, b, c, d, x_nodes) for xi in xx])
    error = np.abs(yy - yy_all)
    print(f"\n==== {k} вузлів ====")
    print(f"Максимальна похибка: {np.max(error):.4f}")
    print(f"Середня похибка: {np.mean(error):.4f}")
    return yy, error

# Тестуємо для 10, 15 та 20 вузлів
yy_10, err_10 = test_nodes(10)
yy_15, err_15 = test_nodes(15)
yy_20, err_20 = test_nodes(20)

# 5. ГРАФІКИ
plt.figure(figsize=(10, 6))
plt.plot(distances, elevations, 'o', label='Вузли (GPS точки)', markersize=5)
plt.plot(xx, yy_all, 'r-', label='Кубічний сплайн (21 вузол)')
plt.plot(xx, yy_10, 'g--', label='10 вузлів')
plt.plot(xx, yy_15, 'b--', label='15 вузлів')
plt.plot(xx, yy_20, 'm--', label='20 вузлів')
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Висота (м)')
plt.title('Профіль висоти маршруту: інтерполяція кубічними сплайнами')
plt.legend()
plt.grid(True)
plt.show()

# Графік похибки для різної кількості вузлів
plt.figure(figsize=(10, 4))
plt.plot(xx, err_10, label='10 вузлів', alpha=0.7)
plt.plot(xx, err_15, label='15 вузлів', alpha=0.7)
plt.plot(xx, err_20, label='20 вузлів', alpha=0.7)
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Похибка (м)')
plt.title('Похибка апроксимації відносно сплайна з 21 вузлом')
plt.legend()
plt.grid(True)
plt.show()

#(ДОДАТКОВО) ХАРАКТЕРИСТИКИ МАРШРУТУ
print("\n" + "="*60)
print("ХАРАКТЕРИСТИКИ МАРШРУТУ")
print("="*60)
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")

# Набір висоти та спуск (по вузлах)
total_ascent = 0
total_descent = 0
for i in range(1, n):
    diff = elevations[i] - elevations[i-1]
    if diff > 0:
        total_ascent += diff
    else:
        total_descent -= diff
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
print(f"Сумарний спуск (м): {total_descent:.2f}")

# Градієнт через похідну сплайна
def spline_derivative(xi, a, b, c, d, x_nodes):
    for i in range(len(x_nodes)-1):
        if x_nodes[i] <= xi <= x_nodes[i+1]:
            dx = xi - x_nodes[i]
            return b[i] + 2*c[i]*dx + 3*d[i]*dx**2
    return 0

grad_full = np.array([spline_derivative(xi, a_all, b_all, c_all, d_all, x_all) for xi in xx]) * 100
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

# Ділянки з крутизною > 15%
steep_segments = np.sum(np.abs(grad_full) > 15)
print(f"Кількість точок з градієнтом > 15%: {steep_segments} (з {len(grad_full)})")

# Механічна енергія підйому
mass = 80  # кг
g = 9.81   # м/с^2
energy = mass * g * total_ascent
print(f"\nМеханічна робота (Дж): {energy:.2f}")
print(f"Механічна робота (кДж): {energy/1000:.2f}")
print(f"Енергія (ккал): {energy/4184:.2f}")

#ДОДАТКОВІ ГРАФІКИ: ГРАДІЄНТ ТА ЕНЕРГІЯ
plt.figure(figsize=(10, 4))
plt.plot(xx, grad_full, color='purple')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Градієнт (%)')
plt.title('Градієнт маршруту (похідна сплайна)')
plt.grid(True)
plt.show()

# Кумулятивна енергія підйому
cumulative_energy = []
current = 0
for i in range(1, len(xx)):
    dh = yy_all[i] - yy_all[i-1]
    if dh > 0: #різниця висот
        current += mass * g * dh
    cumulative_energy.append(current)

plt.figure(figsize=(10, 4))
plt.plot(xx[1:], cumulative_energy, color='orange')
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Енергія (Дж)')
plt.title('Накопичена механічна робота підйому (маса 80 кг)')
plt.grid(True)
plt.show()
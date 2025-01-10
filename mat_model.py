import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m0 = 10000
M = m0 + 24000
Ft = 1609220
Cf = 0.2
ro = 1.225
S = 18.4
g = 9.80665
k = (M - m0) / (3 * 60 + 90)

def mass(t):
    return M - k * t

def acceleration(v, m):
    drag = 0.5 * Cf * ro * v**2 * S
    return (Ft - m * g - drag) / m

def equations(t, y):
    v, h = y
    m = mass(t)
    a = acceleration(v, m)
    return [a, v]

y0 = [0, 0]
t_span = (0, 45)
t_eval = np.arange(0, 45, 0.1)

solution = solve_ivp(equations, t_span, y0, t_eval=t_eval)

times = solution.t
velocities = solution.y[0]
heights = solution.y[1]
masses = mass(times)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(times, velocities, label="Скорость", color="blue")
plt.xlabel("Время (с)")
plt.ylabel("Скорость (м/с)")
plt.title("Скорость от времени")
plt.grid()
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(times, heights, label="Высота", color="green")
plt.xlabel("Время (с)")
plt.ylabel("Высота (м)")
plt.title("Высота от времени")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Гравитационная постоянная и параметры Земли
G = 6.67430e-11  # м^3/(кг*с^2)
M_earth = 5.972e24  # кг
R_earth = 6371000  # м

# Характеристики ракеты
m_initial = 287000  # кг, начальная масса
m_payload = 4725  # кг, масса полезной нагрузки

# Характеристики ступеней ракеты
stages = [
    {"thrust": 20800000, "Isp": 256, "fuel_mass": 200000, "structural_mass": 13000},
    {"thrust": 8000000, "Isp": 3087, "fuel_mass": 50000, "structural_mass": 4000},
    {"thrust": 3000000, "Isp": 3170, "fuel_mass": 10000, "structural_mass": 1000},
]

# Функции расчета

def air_density(h):
    """Возвращает плотность воздуха на высоте h в кг/м^3."""
    if h < 11000:  # тропосфера
        return 1.225 * (1 - 0.000022558 * h) ** 4.256
    elif h < 25000:  # нижняя стратосфера
        return 0.36391 * np.exp(-0.0001577 * (h - 11000))
    else:  # выше стратосферы
        return 0

def gravitational_acceleration(h):
    """Возвращает ускорение свободного падения на высоте h."""
    return G * M_earth / (R_earth + h)**2

def current_stage(m):
    """Определяет текущую ступень ракеты в зависимости от массы."""
    stage_mass = m_initial
    for stage in stages:
        stage_mass -= stage["fuel_mass"] + stage["structural_mass"]
        if m > stage_mass:
            return stage
    return None

def rocket_equations(t, state):
    """Уравнения движения ракеты."""
    h, v, m = state
    stage = current_stage(m)

    if stage and m > m_payload:  # Пока есть топливо
        F_thrust = stage["thrust"]
        burn_rate = F_thrust / (stage["Isp"] * 9.81)
        dm_dt = -burn_rate
    else:  # Топливо закончилось
        F_thrust = 0
        dm_dt = 0

    # Масса ступени отбрасывается
    stage_mass = m_initial
    for s in stages:
        stage_mass -= s["fuel_mass"] + s["structural_mass"]
        if m == stage_mass + s["structural_mass"]:  # Отделение ступени
            m -= s["structural_mass"]

    F_gravity = m * gravitational_acceleration(h)
    rho = air_density(h)
    F_drag = 0.5 * rho * v**2 * 0.3 * (10.3**2 * np.pi / 4)  # Сопротивление воздуха

    dv_dt = (F_thrust - F_gravity - F_drag) / m
    dh_dt = v

    return [dh_dt, dv_dt, dm_dt]

# Начальные условия
h0 = 0  # начальная высота, м
v0 = 0  # начальная скорость, м/с
state0 = [h0, v0, m_initial]

# Временной интервал
t_max = 600  # сек
dt = 0.1  # шаг времени

# Интеграция уравнений
states = [state0]
times = [0]

def integrate(state, dt):
    """Метод Эйлера для численного интегрирования."""
    derivatives = rocket_equations(0, state)
    return [state[i] + derivatives[i] * dt for i in range(3)]

while times[-1] < t_max and states[-1][0] >= 0:
    next_state = integrate(states[-1], dt)
    times.append(times[-1] + dt)
    states.append(next_state)

# Преобразование в массивы для анализа
states = np.array(states)
heights = states[:, 0]
speeds = states[:, 1]
masses = states[:, 2]

# Построение графиков
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(times, heights / 1000)
plt.xlabel('Время (с)')
plt.ylabel('Высота (км)')
plt.title('Высота ракеты')

plt.subplot(3, 1, 2)
plt.plot(times, speeds)
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.title('Скорость ракеты')

plt.subplot(3, 1, 3)
plt.plot(times, masses / 1000)
plt.xlabel('Время (с)')
plt.ylabel('Масса (т)')
plt.title('Масса ракеты')

plt.tight_layout()
plt.show()

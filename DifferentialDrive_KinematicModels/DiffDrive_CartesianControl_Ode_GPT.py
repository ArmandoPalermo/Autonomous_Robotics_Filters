import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ======================
# Simulation setup
# ======================
T = 0.05 # Sampling time [s]
t_final = 8 # Simulation time [s]
# Control gains
k1 = 1.0 # Gain on linear velocity
k2 = 2.0 # Gain on angular velocity
# Initial pose: [x, y, theta]
pose_0 = np.array([1.0, 2.0, np.pi / 2])


# ======================
# Dynamics definition
# ======================
def diff_drive_dynamics(t, pose):
    x, y, theta = pose
    e_x, e_y, e_theta = pose
    # Control law
    v = -k1 * (e_x * np.cos(e_theta) + e_y * np.sin(e_theta))
    w = k2 * (np.arctan2(e_y, e_x) - e_theta + np.pi)
    dxdt = v * np.cos(theta)
    dydt = v * np.sin(theta)
    dthetadt = w
    return [dxdt, dydt, dthetadt]


# ======================
# Integration with solve_ivp
# ======================
sol = solve_ivp(
    diff_drive_dynamics,
    [0, t_final],
    pose_0,
    t_eval=np.arange(0, t_final, T),
    method="RK45"
    )
x, y, theta = sol.y
time = sol.t


# ======================
# Plots
# ======================
plt.figure(figsize=(6, 6))
plt.plot(x, y, '-*', label='Robot poses')
plt.plot(pose_0[0], pose_0[1], 'dr', label='Initial pose')
plt.plot(x[-1], y[-1], 'or', label='Final pose')
plt.title("Mobile robot position in the plane (solve_ivp)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.axis("equal")
plt.grid(True)


plt.figure(figsize=(8, 4))
plt.plot(time, x, label="x coordinate", linewidth=2)
plt.plot(time, y, label="y coordinate", linewidth=2)
plt.plot(time, theta, label="theta coordinate", linewidth=2)
plt.title("Mobile robot pose evolution (solve_ivp)")
plt.xlabel("Time [s]")
plt.ylabel("Pose [m/rad]")

plt.show()
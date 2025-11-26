import numpy as np
import matplotlib.pyplot as plt

# ======================
# Simulation setup
# ======================
T = 0.05 # Sampling time [s]
t_final = 8 # Simulation time [s]
n_samples = int(np.ceil(t_final / T))
time = np.arange(1, n_samples + 1) * T

# Control gains
k1 = 1.0 # Gain on linear velocity
k2 = 2.0 # Gain on angular velocity

# Initial pose: [x, y, theta]
pose_0 = np.array([1.0, 2.0, np.pi / 2])
# Pose evolution (each column: [x, y, theta]^T at time step)
pose = np.zeros((3, n_samples))
pose[:, 0] = pose_0


# ======================
# Robot kinematics
# ======================
def kinematic_diff_drive(pose, u, T):
    x, y, theta = pose
    v, w = u
    x_next = x + T * v * np.cos(theta)
    y_next = y + T * v * np.sin(theta)
    theta_next = theta + T * w
    return np.array([x_next, y_next, theta_next])


# ======================
# Simulation loop
# ======================
for t in range(n_samples - 1):
    pose_t = pose[:, t]
    e_x, e_y, e_theta = pose_t
    v_t = -k1 * (e_x * np.cos(e_theta) + e_y * np.sin(e_theta))
    w_t = k2 * (np.arctan2(e_y, e_x) - e_theta + np.pi)
    u_t = np.array([v_t, w_t])
    pose[:, t + 1] = kinematic_diff_drive(pose_t, u_t, T)


# ======================
# Plots
# ======================
x, y, theta = pose

plt.figure(figsize=(6, 6))
plt.plot(x, y, '-*', label='Robot poses')
plt.plot(pose_0[0], pose_0[1], 'dr', label='Initial pose')
plt.plot(x[-1], y[-1], 'or', label='Final pose')
plt.title("Mobile robot position in the plane")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.axis("equal")
plt.grid(True)

plt.figure(figsize=(8, 4))
plt.plot(time, x, label="x coordinate", linewidth=2)
plt.plot(time, y, label="y coordinate", linewidth=2)
plt.plot(time, theta, label="theta coordinate", linewidth=2)
plt.title("Mobile robot pose evolution")
plt.xlabel("Time [s]")
plt.ylabel("Pose [m/rad]")
plt.legend()
plt.grid(True)

plt.show()

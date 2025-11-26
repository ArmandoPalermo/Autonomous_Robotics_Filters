#
#   Kinematic robot model
#   Mobile Robots/Autonomous Robotics
#   Paolo Valigi
#   University of Perugia
#
#   Version 1.0, September 2025
#   Based on previous matlab code, rewritten in python by chat-GPT
#
#

import numpy as np
import matplotlib.pyplot as plt
from dask.dataframe.dispatch import tolist
from matplotlib.animation import FuncAnimation

# ======================
# Simulation setup
# ======================
T = 0.05  # Sampling time [s]
t_final = 12  # Simulation time [s]
n_samples = int(np.ceil(t_final / T))
time = np.arange(1, n_samples + 1) * T

# Initial pose: [x, y, theta]
pose_0 = np.array([1.0, 2.0, np.pi / 2])
# Pose evolution (each column: [x, y, theta]^T at time step)
poses = np.zeros((3, n_samples))
poses[:, 0] = pose_0


# ======================
# Robot kinematics
# ======================
def motion_model_velocity(pose: np.ndarray, u: np.ndarray, T: float):
    """
    pose: np.ndarray = [x, y, theta] components of the current pose vector
    u: np.ndarray = [v, w]   (driving velocity, angular velocity)
    T: float = control/integration  period
    """

    x_c, y_c, theta_c = pose # current components of pose vector
    v, w = u

    if abs(w) > 1e-10:  # circular motion
        r = v/w
        x_next = x_c - r*np.sin(theta_c) + r*np.sin(theta_c + w*T)
        y_next = y_c + r*np.cos(theta_c) - r*np.cos(theta_c + w*T)
        theta_next = theta_c + T * w
    else:  # linear motion
        x_next = x_c + v*T*np.cos(theta_c)
        y_next = y_c + v*T*np.sin(theta_c)
        theta_next = theta_c

    return np.array([x_next, y_next, theta_next])


# ======================
# Simulation loop
# ======================
for t in range(n_samples - 1):
    pose_t = poses[:, t]
    v_t = 1
    w_t = 0.5
    u_t = np.array([v_t, w_t])
    poses[:, t + 1] = motion_model_velocity(pose_t, u_t, T)


# ======================
# Plots
# ======================
x, y, theta = poses

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

# plt.show()


# Animation code, by chat-GPT
xs, ys, thetas = poses
# -----------------------------
# matplotlib animation
# -----------------------------
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlim(min(xs)-1, max(xs)+1)
ax.set_ylim(min(ys)-1, max(ys)+1)

trajectory_line, = ax.plot([], [], 'b-', lw=2, label="Trajectory")
robot_point, = ax.plot([], [], 'ro', markersize=8, label="Robot")
heading_line, = ax.plot([], [], 'r-', lw=2)


def init():
    trajectory_line.set_data([], [])
    robot_point.set_data([], [])
    heading_line.set_data([], [])
    return trajectory_line, robot_point, heading_line


def update(frame):
    trajectory_line.set_data(xs[:frame], ys[:frame])
    robot_point.set_data([xs[frame]], [ys[frame]])

    dx = 0.5 * np.cos(thetas[frame])
    dy = 0.5 * np.sin(thetas[frame])
    heading_line.set_data([xs[frame], xs[frame] + dx],
                          [ys[frame], ys[frame] + dy])
    return trajectory_line, robot_point, heading_line


animation = FuncAnimation(fig, update, frames=len(xs), init_func=init,
                          blit=True, interval=50, repeat=False)

plt.legend()
plt.show()

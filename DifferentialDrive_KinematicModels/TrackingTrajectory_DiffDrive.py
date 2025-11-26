import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation

from KinematicsModels_Confronto import eulermodel,rkmodel,velocity_control_model

if __name__ == '__main__':

    posa_iniziale = [0, 0, 0]
    b = 0.1
    kx = 2
    ky = 2

    period = 0.05
    t_finale = 25
    numpassi = int(np.ceil(t_finale / period))
    pose_model = [posa_iniziale]



    for t in range(numpassi):
        matrice_disaccoppiata = np.vstack([[np.cos(posa_iniziale[2]), - b * np.sin(posa_iniziale[2])],
                                           [np.sin(posa_iniziale[2]), b * np.cos(posa_iniziale[2])]])

        mat_inv = np.linalg.inv(matrice_disaccoppiata)
        tempo = t * period
        xr_b = 0.5 * tempo  # moto lineare lungo x
        yr_b = 1.5 * np.sin(2.0 * tempo)  # sinusoide lungo y

        xr_b_1 = 0.5  # derivata costante
        yr_b_1 = 1.5 * 2.0 * np.cos(2.0 * tempo)  # derivata della sinusoide

        mat_control_scheme = np.vstack([-kx * (posa_iniziale[0] + b * np.cos(posa_iniziale[2]) - xr_b) + xr_b_1,
                                        -ky * (posa_iniziale[1] + b * np.sin(posa_iniziale[2]) - yr_b) + yr_b_1])

        [[v],[w]] = np.dot(mat_inv,mat_control_scheme)

        posa_calc = velocity_control_model(posa_iniziale, v, w, period)
        pose_model.append(posa_calc)
        posa_iniziale = pose_model[-1]


    pose_arr_model = np.array(pose_model)
    x_k = pose_arr_model[:,0]
    y_k = pose_arr_model[:,1]
    theta_k = pose_arr_model[:,2]

    # Traiettoria desiderata del punto b
    t_vals = np.linspace(0, t_finale, numpassi)
    xr_vals = 0.5 * t_vals
    yr_vals = 1.5 * np.sin(2.0 * t_vals)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot(xr_vals, yr_vals, '--', color='gray', label='Traiettoria desiderata')
    ax.legend()
    ax.set_xlim(min(min(xr_vals), min(x_k)) - 1, max(max(xr_vals), max(x_k)) + 1)
    ax.set_ylim(min(min(yr_vals), min(y_k)) - 1, max(max(yr_vals), max(y_k)) + 1)

    robot_path, = ax.plot([], [], 'r-', lw=2)
    robot_point, = ax.plot([], [], 'ro')
    orientation_line, = ax.plot([], [], 'b-', lw=2)


    def animate(i):
        robot_path.set_data(x_k[:i + 1], y_k[:i + 1])
        robot_point.set_data([x_k[i]], [y_k[i]])
        L = 0.5
        xh = x_k[i] + L * np.cos(theta_k[i])
        yh = y_k[i] + L * np.sin(theta_k[i])
        orientation_line.set_data([x_k[i], xh], [y_k[i], yh])
        return robot_path, robot_point, orientation_line


    ani = animation.FuncAnimation(fig, animate, frames=len(x_k), blit=True, interval=50)
    plt.show()
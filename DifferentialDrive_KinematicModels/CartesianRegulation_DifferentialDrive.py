import numpy as np
import matplotlib.pyplot as plt
import math
from KinematicsModels_Confronto import eulermodel,rkmodel,velocity_control_model





if __name__ == '__main__':

    posa_iniziale = [2, 1, 0]
    posa_desiderata = [8, 1, 0]
    #def parametri algoritmo di controllo
    k1 = 1
    k2 = 2
    k3 = 0.8
    period = 0.05
    t_finale = 25
    numpassi = int(np.ceil(t_finale/period))
    #inizializzo un array vuoto e aggiungo la posa iniziale
    pose_model = [posa_iniziale]


    for i in range(numpassi):

        x = posa_iniziale[0]
        y = posa_iniziale[1]
        theta = posa_iniziale[2]

        xd = posa_desiderata[0]
        yd = posa_desiderata[1]
        thetad = posa_desiderata[2]

        #Leggi di controllo senza orientamento
        #v_k = -k1 * ((x - xd) * np.cos(theta) +  (y - yd) * np.sin(theta))
        #w_k = k2 * (np.arctan2((y - yd),(x - xd)) - theta + np.pi)

        #Legge di controllo con orientamento
        ro = np.sqrt(np.power((x - xd),2) + np.power((y - yd),2))
        gamma = np.arctan2((y - yd),(x - xd)) - (theta - thetad) + np.pi
        delta = (theta - thetad) + gamma

        v_k = k1 * ro * np.cos(gamma)
        w_k = k2 * gamma + (k1 * (np.sin(gamma) * np.cos(gamma))/gamma) * (gamma + k3 * delta)

        posa_calc = eulermodel(posa_iniziale, v_k, w_k, period)
        pose_model.append(posa_calc)
        posa_iniziale = pose_model[-1]

    pose_arr_model = np.array(pose_model)
    print(pose_arr_model[-1])
    x_k = pose_arr_model[:,0]
    y_k = pose_arr_model[:,1]
    theta_k = pose_arr_model[:,2]

    plt.plot(x_k,y_k, color = 'red')
    plt.show()

    time = np.arange(1, numpassi + 2) * period
    plt.figure(figsize=(8, 4))
    plt.plot(time, x_k, label="x coordinate", linewidth=2)
    plt.plot(time, y_k, label="y coordinate", linewidth=2)
    plt.plot(time, theta_k, label="theta coordinate", linewidth=2)
    plt.title("Mobile robot pose evolution")
    plt.xlabel("Time [s]")
    plt.ylabel("Pose [m/rad]")
    plt.legend()
    plt.grid(True)

    plt.show()

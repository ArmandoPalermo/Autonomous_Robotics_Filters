import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from matplotlib.lines import lineStyles


def eulermodel(posa_prec ,v_k, w_k, period):
    xkp1 = posa_prec[0] + v_k * period * math.cos(posa_prec[2])
    ykp1 = posa_prec[1] + v_k * period * math.sin(posa_prec[2])
    thetakp1 = posa_prec[2] + w_k * period
    return  [xkp1,ykp1,thetakp1]


def rkmodel(posa_prec, v_k, w_k, period):
    xkp1 = posa_prec[0] + v_k * period * math.cos(posa_prec[2] + (w_k * period)/2)
    ykp1 = posa_prec[1] + v_k * period * math.sin(posa_prec[2] + (w_k * period)/2)
    thetakp1 = posa_prec[2] + w_k * period
    return [xkp1, ykp1, thetakp1]

def velocity_control_model(posa_prec, v_k, w_k, period):
    xkp1 = posa_prec[0] - (v_k/w_k) * math.sin(posa_prec[2]) + (v_k/w_k) * math.sin(posa_prec[2] + w_k * period)
    ykp1 = posa_prec[1] + (v_k/w_k) * math.cos(posa_prec[2]) - (v_k/w_k) * math.cos(posa_prec[2] + w_k * period)
    thetakp1 = posa_prec[2] + w_k * period
    return [xkp1, ykp1, thetakp1]


if __name__ == '__main__':
    period = 0.1
    t_finale = 15
    numpassi = int(np.ceil(t_finale/period))
    posa_iniziale = [0,0,0]

    #inizializzo un array vuoto e aggiungo la posa iniziale
    poseeuler = [posa_iniziale]
    poserk = [posa_iniziale]
    pose_vcm = [posa_iniziale]

    #legge di controllo della velocità
    v_k = 1.5
    w_k = 0.5

    #Calcolo delle pose con metodo di eulero
    for i in range(numpassi):
        #Provo a rendere non costanti le velocita(fai altri test)


        posa_calc = eulermodel(poseeuler[i],v_k,w_k,period)
        poseeuler.append(posa_calc)
        posa_calc = rkmodel(poserk[i], v_k, w_k, period)
        poserk.append(posa_calc)
        posa_calc = velocity_control_model(pose_vcm[i], v_k, w_k, period)
        pose_vcm.append(posa_calc)

    #plot delle pose di eulero in blu
    pose_array_euler = np.array(poseeuler)
    x = pose_array_euler[:, 0]
    y = pose_array_euler[:, 1]
    theta = pose_array_euler[:, 2]
    plt.plot(x,y, color='blue')

    # plot delle pose di rk in rosso
    pose_array_rk = np.array(poserk)
    x_rk = pose_array_rk[:, 0]
    y_rk = pose_array_rk[:, 1]
    theta_rk = pose_array_rk[:, 2]

    plt.plot(x_rk,y_rk, color='red')


    # plot delle pose di vcm in verde tratteggiato
    pose_array_vcm = np.array(pose_vcm)
    x_vcm = pose_array_vcm[:, 0]
    y_vcm = pose_array_vcm[:, 1]
    theta_vcm = pose_array_vcm[:, 2]

    plt.plot(x_vcm,y_vcm, color='green', linestyle ='dashdot')
    plt.show()




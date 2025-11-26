import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import math

from matplotlib.lines import lineStyles


def eulermodel(posa_prec ,v_k, w_k, gamma_1, period):
    xkp1 = posa_prec[0] + v_k * period * math.cos(posa_prec[2])
    ykp1 = posa_prec[1] + v_k * period * math.sin(posa_prec[2])
    #Aggiunta termine stocastico
    thetakp1 = posa_prec[2] + w_k * period + gamma_1 * period
    return  [xkp1,ykp1,thetakp1]


def rkmodel(posa_prec, v_k, w_k, gamma_1, period):
    xkp1 = posa_prec[0] + v_k * period * math.cos(posa_prec[2] + (w_k * period)/2)
    ykp1 = posa_prec[1] + v_k * period * math.sin(posa_prec[2] + (w_k * period)/2)
    # Aggiunta termine stocastico
    thetakp1 = posa_prec[2] + w_k * period + gamma_1 * period
    return [xkp1, ykp1, thetakp1]

def velocity_control_model(posa_prec, v_k, w_k, gamma_1, period):
    xkp1 = posa_prec[0] - (v_k/w_k) * math.sin(posa_prec[2]) + (v_k/w_k) * math.sin(posa_prec[2] + w_k * period)
    ykp1 = posa_prec[1] + (v_k/w_k) * math.cos(posa_prec[2]) - (v_k/w_k) * math.cos(posa_prec[2] + w_k * period)
    # Aggiunta termine stocastico
    thetakp1 = posa_prec[2] + w_k * period + gamma_1 * period
    return [xkp1, ykp1, thetakp1]

def calc_input_stochastic(theta,period, v_k,w_k,alphaParameters):
    #Calcolando le varianze del rumore uniforme dell'input
    sigmav = math.sqrt(alphaParameters[0] * math.pow(v_k,2) + alphaParameters[1] * math.pow(w_k,2))
    sigmaw = math.sqrt(alphaParameters[2] * math.pow(v_k,2) + alphaParameters[3] * math.pow(w_k,2))
    sigma_theta = math.sqrt(alphaParameters[4] * math.pow(v_k,2) + alphaParameters[5] * math.pow(w_k,2))

    v_k_1 = v_k + rnd.gauss(0,sigmav)
    w_k_1 = w_k + rnd.gauss(0, sigmaw)
    gamma_1 = rnd.gauss(0,sigma_theta)
    return [v_k_1,w_k_1,gamma_1]

def distanza_finale(pose_noise, pose_deterministico):
    dx = pose_noise[-1][0] - pose_deterministico[-1][0]
    dy = pose_noise[-1][1] - pose_deterministico[-1][1]
    return np.sqrt(dx**2 + dy**2)



period = 0.1
t_finale = 15
numpassi = int(np.ceil(t_finale/period))
posa_iniziale = [0,0,0]

#inizializzo un array vuoto e aggiungo la posa iniziale
poseeuler = [posa_iniziale]
poseeuler_noise = [posa_iniziale]
poserk = [posa_iniziale]
poserk_noise = [posa_iniziale]
pose_vcm = [posa_iniziale]
pose_vcm_noise = [posa_iniziale]

#legge di controllo della velocità
v_k = 1.5
w_k = 0.5

#Parametri per modelli stocastici
alphaParameters = [0, 0 , 0, 0, 0.5, 0]
nullAlphaParameters = [0, 0, 0, 0, 0, 0]
#Calcolo delle pose con metodo di eulero
n = 0
pose_finali_euler_x = []
pose_finali_euler_y = []
pose_finali_rk_x = []
pose_finali_rk_y = []
pose_finali_vcm_x = []
pose_finali_vcm_y = []





for i in range(numpassi):
    #Euler con rumore
    [v_k_1, w_k_1, gamma_1] = calc_input_stochastic(poseeuler_noise[i][2], period, v_k, w_k, alphaParameters)
    posa_calc = eulermodel(poseeuler_noise[i],v_k_1,w_k_1,gamma_1,period)
    poseeuler_noise.append(posa_calc)

    #Rk con rumore
    [v_k_1, w_k_1, gamma_1] = calc_input_stochastic(poserk_noise[i][2], period, v_k, w_k, alphaParameters)
    posa_calc = rkmodel(poserk_noise[i], v_k_1, w_k_1, gamma_1,period)
    poserk_noise.append(posa_calc)

    #Velocity motion con rumore
    [v_k_1, w_k_1, gamma_1] = calc_input_stochastic(pose_vcm_noise[i][2], period, v_k, w_k, alphaParameters)
    posa_calc = velocity_control_model(pose_vcm_noise[i], v_k_1, w_k_1, gamma_1, period)
    pose_vcm_noise.append(posa_calc)

    # Euler senza rumore
    [v_k_1, w_k_1, gamma_1] = calc_input_stochastic(poseeuler[i][2], period, v_k, w_k, nullAlphaParameters)
    posa_calc = eulermodel(poseeuler[i], v_k_1, w_k_1, gamma_1, period)
    poseeuler.append(posa_calc)

    # RK2 senza rumore
    [v_k_1, w_k_1, gamma_1] = calc_input_stochastic(poserk[i][2], period, v_k, w_k, nullAlphaParameters)
    posa_calc = rkmodel(poserk[i], v_k_1, w_k_1, gamma_1, period)
    poserk.append(posa_calc)

    # VCM senza rumore
    [v_k_1, w_k_1, gamma_1] = calc_input_stochastic(pose_vcm[i][2], period, v_k, w_k, nullAlphaParameters)
    posa_calc = velocity_control_model(pose_vcm[i], v_k_1, w_k_1, gamma_1, period)
    pose_vcm.append(posa_calc)


# Converto le pose in array NumPy
pose_array_euler = np.array(poseeuler)
pose_array_euler_noise = np.array(poseeuler_noise)
pose_array_rk = np.array(poserk)
pose_array_rk_noise = np.array(poserk_noise)
pose_array_vcm = np.array(pose_vcm)
pose_array_vcm_noise = np.array(pose_vcm_noise)

# Plot Euler: deterministico (blu continuo) vs stocastico (blu tratteggiato)
plt.plot(pose_array_euler[:, 0], pose_array_euler[:, 1], color='blue', linestyle='solid', label='Euler deterministico')
plt.plot(pose_array_euler_noise[:, 0], pose_array_euler_noise[:, 1], color='blue', linestyle='dashed', label='Euler stocastico')

# Plot RK: deterministico (rosso continuo) vs stocastico (rosso tratteggiato)
plt.plot(pose_array_rk[:, 0], pose_array_rk[:, 1], color='red', linestyle='solid', label='RK deterministico')
plt.plot(pose_array_rk_noise[:, 0], pose_array_rk_noise[:, 1], color='red', linestyle='dashed', label='RK stocastico')

# Plot VCM: deterministico (verde continuo) vs stocastico (verde tratteggiato)
plt.plot(pose_array_vcm[:, 0], pose_array_vcm[:, 1], color='green', linestyle='solid', label='VCM deterministico')
plt.plot(pose_array_vcm_noise[:, 0], pose_array_vcm_noise[:, 1], color='green', linestyle='dashed', label='VCM stocastico')

# Aggiungo legenda e titolo
plt.legend()
plt.title('Confronto traiettorie: deterministiche vs stocastiche')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.grid(True)
plt.axis('equal')
plt.show()


#Istogramma delle distanze tra le pose dei modelli con rumore e senza
plt.bar(['Euler', 'RK', 'VCM'], [
    distanza_finale(poseeuler_noise, poseeuler),
    distanza_finale(poserk_noise, poserk),
    distanza_finale(pose_vcm_noise, pose_vcm)
])
plt.title('Deviazione finale tra modelli stocastici e deterministici')
plt.ylabel('Distanza [m]')
plt.grid(True)
plt.show()




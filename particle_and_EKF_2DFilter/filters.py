import numpy as np
from models import *

def particle_filter(t,true_position,v_t_nominale,alpha_v,alpha_theta,sigma_z,beacons,particles_array_x,particles_array_y,particles_weight):
    # Aggiorno la posizione del robot
    true_position[0], true_position[1], v_t, theta_t = motion_model(t, true_position[0], true_position[1], v_t_nominale,
                                                                    alpha_v, alpha_theta)

    # Per ogni beacon devo calcolare la distanza dal robot
    true_meas = np.zeros(len(beacons))
    for b in range(len(beacons)):
        true_meas[b] = np.linalg.norm(np.subtract(true_position, beacons[b])) + np.random.normal(0, sigma_z)

    weight_tot = 0
    # Ciclo per ogni particella
    for p in range(len(particles_array_x)):
        likelihood = 1
        particles_array_x[p], particles_array_y[p], _, _ = motion_model(t, particles_array_x[p], particles_array_y[p],
                                                                        v_t_nominale, alpha_v, alpha_theta)
        particles_pos = np.array([particles_array_x[p], particles_array_y[p]])
        # Mi calcolo la misurazione tramite il modello di misura da ogni particella verso i beacon(likelihood)
        for b in range(len(beacons)):
            likelihood = likelihood * measurement_model(true_meas[b], particles_pos, beacons[b], sigma_z)[1]
        particles_weight[p] = likelihood
        weight_tot += particles_weight[p]

    # Normalizzazione delle particelle
    # Gestisco la divisione per zero nel caso in cui il peso delle particelle sia nullo
    # ovvero il caso in cui le particelle si trovano lontane dalla vera posizione e quindi la verosimiglianza
    # della misura è nulla
    if weight_tot > 0:
        particles_weight /= weight_tot
    else:
        particles_weight = np.ones(len(particles_weight)) / len(particles_weight)

    # Calcolo della stima con la moda
    idx = np.argmax(particles_weight)
    x_stima = particles_array_x[idx]
    y_stima = particles_array_y[idx]

    # CDF della distribuzione delle particelle
    cumulative_sum = np.cumsum(particles_weight)

    # Resampling, prendo un numero a caso tra 0 e 1 , e con searchsorted prendo il valore della particella con quel peso
    for p in range(len(particles_array_x)):
        r = np.random.uniform(0, 1)
        # Utile quando ho una distribuzione crescente, mi dice a quale x corrisponde la y(in questo caso r)
        idx = np.searchsorted(cumulative_sum, r)
        pos_x = particles_array_x[idx]
        pos_y = particles_array_y[idx]  # posizione della particella scelta
        particles_array_x[p] = pos_x
        particles_array_y[p] = pos_y



    #peso uniforme delle particelle
    particles_weight[:] = np.ones(len(particles_weight)) / len(particles_weight)

    return true_position, v_t, x_stima,y_stima,particles_array_x,particles_array_y, particles_weight


def kalman_filter(t,true_position,x_stima,v_t_nominale,alpha_v,alpha_theta,sigma_z,beacons,prior,posterior):
    # Sposto il robot reale con il motion model
    true_position[0], true_position[1], v_t, theta_t = motion_model(
        t, true_position[0], true_position[1], v_t_nominale, alpha_v, alpha_theta
    )

    # Per ogni beacon devo calcolare la distanza dal robot
    true_meas = np.zeros(len(beacons))
    for b in range(len(beacons)):
        true_meas[b] = np.linalg.norm(np.subtract(true_position, beacons[b])) + np.random.normal(0, sigma_z)

    # --- Predizione coerente con il modello reale ---
    x_pred_x, x_pred_y, v_pred, theta_pred = motion_model(
        t, x_stima[0], x_stima[1], v_t_nominale, 0, 0
    )
    x_pred = np.array([x_pred_x, x_pred_y])

    # Fx è una matrice identità diagonale 2x2
    jacobian_fx = np.eye(2)
    # Fu è ottenuto tramite la derivazione dello stato rispetto all'ingresso, u = [v_t,theta]
    jacobian_fu = np.array([
        [np.cos(theta_pred), -v_pred * np.sin(theta_pred)],
        [np.sin(theta_pred), v_pred * np.cos(theta_pred)]
    ])

    # Il valore assoluto qui non è necessario, ma è per tenere in considerazione che la velocità può essere negativa
    # Quindi se per qualche motivo si cambia la varianza almeno è positiva
    q_t = np.diag((alpha_v * np.abs(v_pred) ** 2, alpha_theta * np.abs(v_pred) ** 2))
    r_t = np.eye(len(beacons)) * sigma_z ** 2

    # Calcolo la prior sulla base delle matrici appena calcolate
    prior = np.linalg.multi_dot([jacobian_fx, posterior, np.transpose(jacobian_fx)]) + np.linalg.multi_dot(
        [jacobian_fu, q_t, np.transpose(jacobian_fu)]
    )


    # Calcolo z_hat e Hx
    z_hat = np.zeros(len(beacons))
    Hx = np.zeros((len(beacons), 2))
    for b in range(len(beacons)):
        z_hat[b] = np.linalg.norm(np.subtract(x_pred, beacons[b]))
        if z_hat[b] < 1e-8:  # Evita divisione per zero
            z_hat[b] = 1e-8
        Hx[b, 0] = (x_pred[0] - beacons[b][0]) / z_hat[b]
        Hx[b, 1] = (x_pred[1] - beacons[b][1]) / z_hat[b]

    # Correzione tramite misura
    innovation = true_meas - z_hat
    s_t = np.linalg.multi_dot([Hx, prior, np.transpose(Hx)]) + r_t
    if np.linalg.cond(s_t) > 1e12:  # Protezione numerica
        s_t += 1e-6 * np.eye(s_t.shape[0])
    l_t = np.linalg.multi_dot([prior, np.transpose(Hx), np.linalg.inv(s_t)])

    # Correzione
    x_stima = x_pred + np.dot(l_t, innovation)
    posterior = (np.eye(2) - l_t @ Hx) @ prior

    errore = np.linalg.norm(true_position - x_stima)
    v_t_nominale = v_pred

    return true_position, x_stima, v_t_nominale, prior, posterior, errore

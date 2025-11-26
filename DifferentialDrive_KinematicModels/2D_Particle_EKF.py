import numpy as np
import matplotlib.pyplot as plt



def motion_model(t,true_position_x,true_position_y,v_t_nominale,alpha_v,alpha_theta):

    v_t = v_t_nominale + np.cos(t)
    theta_t = 0.5 * np.sin(0.4 * t)

    #Aggiunta rumore
    v_t = v_t + np.random.normal(0, np.abs(v_t) * np.sqrt(alpha_v))
    theta_t = theta_t + np.random.normal(0, np.abs(v_t) * np.sqrt(alpha_theta))

    # Aggiornamento moto
    true_position_x = true_position_x + v_t * np.cos(theta_t)
    true_position_y = true_position_y + v_t * np.sin(theta_t)
    return true_position_x,true_position_y,v_t,theta_t

def measurement_model(z_att, x_att, m1, sigma_z):
    expected_z = np.linalg.norm(x_att - m1)
    return (1 / np.sqrt(2 * np.pi * sigma_z**2)) * np.exp(-0.5 * ((z_att - expected_z)**2) / sigma_z**2)



USE_PARTICLE_FILTER = False
USE_KALMAN_FILTER = True

if __name__ == "__main__" :
    #Definizione confine Mappa
    x_min,y_min = 0,0
    x_max = 40
    y_max = 40

    #Definizione posizione reale del robot
    true_position = [10,10]
    #Definizione posizione beacon
    m1 = [20,20]
    m2 = [10, 20]
    m3 = [30, 10]


    if USE_PARTICLE_FILTER:
        #Gestione iniziale particelle
        N = 10000
        particles_array_x = np.random.uniform(x_min,x_max,N)
        particles_array_y = np.random.uniform(y_min,y_max,N)
        particles_weight = np.ones(N)/N


        #Ciclo di simulazione
        n_cicli = 20
        v_t_nominale= 1


        #Valore sigma dell'incertezza di misura
        sigma_z = 0.3
        #Coefficienti alpha per la definizione del rumore del motion model
        alpha_v = 0.05
        alpha_theta = 0.05

        plt.figure(figsize=(8, 8))
        plt.grid(True)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('Dimensione x')
        plt.ylabel('Dimensione y')

        for t in range(n_cicli):
            #Definisco la la legge di v_t, in sostanza varia tra 1 e 2 , quindi il robot accelera e decelera
            true_position[0],true_position[1],v_t,theta_t = motion_model(t,true_position[0],true_position[1],v_t_nominale,alpha_v, alpha_theta)
            v_t_nominale = v_t
            print(v_t)
            #Aggiornamento misura reale
            true_z1_t = np.linalg.norm(np.subtract(true_position,m1)) + np.random.normal(0,sigma_z)
            true_z2_t = np.linalg.norm(np.subtract(true_position, m2)) + np.random.normal(0, sigma_z)
            true_z3_t = np.linalg.norm(np.subtract(true_position, m3)) + np.random.normal(0, sigma_z)

            weight_tot = 0
            for i in range(N):
                #Ciclo per tutte le particelle
                particles_array_x[i],particles_array_y[i],_,_ = motion_model(t,particles_array_x[i],particles_array_y[i], v_t_nominale,alpha_v, alpha_theta)
                particles_array_pos = np.array([particles_array_x[i], particles_array_y[i]])
                #Calcolo i pesi delle singole particelle
                mes_model1 = measurement_model(true_z1_t, particles_array_pos, m1, sigma_z)
                mes_model2 = measurement_model(true_z2_t, particles_array_pos, m2, sigma_z)
                mes_model3 = measurement_model(true_z3_t, particles_array_pos, m3, sigma_z)
                particles_weight[i] = mes_model1 * mes_model2 * mes_model3
                weight_tot += particles_weight[i]

            #Aggiornamento delle v nominale
            v_t_nominale = v_t
            #Normalizzazione--> gestita la divisione per 0
            #Se il peso totale è nullo allora assegno valori uniformi ai pesi
            if weight_tot > 0:
                particles_weight /= weight_tot
            else:
                particles_weight[:] = 1.0 / N


            #Resampling
            particles_array_resampled = []

            # CDF della distribuzione delle particelle
            cumulative_sum = np.cumsum(particles_weight)


            plt.scatter(particles_array_x,particles_array_y, color = "red")

            ax = plt.gca()  # Ottieni l'oggetto Axes corrente

            circle1 = plt.Circle((m1[0], m1[1]), true_z1_t, color='blue', fill=False)
            circle2 = plt.Circle((m2[0], m2[1]), true_z2_t, color='blue', fill=False)
            circle3 = plt.Circle((m3[0], m3[1]), true_z3_t, color='blue', fill=False)

            ax.add_artist(circle1)
            ax.add_artist(circle2)
            ax.add_artist(circle3)

            idx = np.argmax(particles_weight)
            x_stima = particles_array_x[idx]
            y_stima = particles_array_y[idx]
            plt.plot(x_stima, y_stima, marker='o', color='black', label='Stima')
            plt.plot(m1[0],m1[1], marker = 'D', color = 'blue', label = 'Beacon m1')
            plt.plot(m2[0], m2[1], marker='D', color='blue', label='Beacon m2')
            plt.plot(m3[0], m3[1], marker='D', color='blue', label='Beacon m3')
            plt.plot(true_position[0], true_position[1], marker='s', color='green', label='TruePos')
            plt.title('Particle Filter 2D')
            plt.axis('square')
            plt.legend()
            plt.show()

            # Resampling, prendo un numero a caso tra 0 e 1 , e con searchsorted prendo il valore della particella con quel peso
            for p in range(N):
                r = np.random.uniform(0, 1)
                # Utile quando ho una distribuzione crescente, mi dice a quale x corrisponde la y(in questo caso r)
                idx = np.searchsorted(cumulative_sum, r)
                pos_x = particles_array_x[idx]
                pos_y = particles_array_y[idx]  # posizione della particella scelta
                particles_array_x[p] = pos_x
                particles_array_y[p] = pos_y

            particles_weight[:] = 1.0 / N

    if USE_KALMAN_FILTER:

        true_position = [0, 0]
        x_est = np.array([0.0, 0.0])

        v_t_nominale = 1
        alpha_v = 0
        alpha_theta = 0
        sigma_z = 5

        # Varianze lungo x e y
        var_x, var_y = 3, 3
        P_0 = np.diag([var_x, var_y])
        P_posterior = P_0

        n_cicli = 20
        num_beacon = 3

        for t in range(n_cicli):
            # Simulazione movimento reale
            true_position[0], true_position[1], v_t, theta_t = motion_model(
                t, true_position[0], true_position[1], v_t_nominale, alpha_v, alpha_theta
            )

            # Predizione EKF
            alpha_v = 0.01
            alpha_theta = 0.01
            q_t = np.diag((alpha_v * v_t ** 2, alpha_theta * v_t ** 2))

            jacobian_fx = np.eye(2)
            jacobian_fu = np.array([
                [np.cos(theta_t), -v_t * np.sin(theta_t)],
                [np.sin(theta_t), v_t * np.cos(theta_t)]
            ])

            prior = (jacobian_fx @ P_posterior @ jacobian_fx.T) + (jacobian_fu @ q_t @ jacobian_fu.T)

            # Predizione della posizione stimata
            x_est = x_est + np.array([v_t * np.cos(theta_t), v_t * np.sin(theta_t)])

            # === Misure rumorose e predette ===
            z = np.array([
                np.linalg.norm(np.subtract(true_position, m1)) + np.random.normal(0, sigma_z),
                np.linalg.norm(np.subtract(true_position, m2)) + np.random.normal(0, sigma_z),
                np.linalg.norm(np.subtract(true_position, m3)) + np.random.normal(0, sigma_z)
            ])

            z_hat = np.array([
                np.linalg.norm(np.subtract(x_est, m1)),
                np.linalg.norm(np.subtract(x_est, m2)),
                np.linalg.norm(np.subtract(x_est, m3))
            ])

            innovation = z - z_hat

            Hx = np.array([
                [(x_est[0] - m1[0]) / z_hat[0], (x_est[1] - m1[1]) / z_hat[0]],
                [(x_est[0] - m2[0]) / z_hat[1], (x_est[1] - m2[1]) / z_hat[1]],
                [(x_est[0] - m3[0]) / z_hat[2], (x_est[1] - m3[1]) / z_hat[2]]
            ])

            r_t = np.eye(num_beacon) * sigma_z ** 2

            s_t = Hx @ prior @ Hx.T + r_t
            l_t = prior @ Hx.T @ np.linalg.inv(s_t)

            #Correzione
            x_est = x_est + l_t @ innovation
            P_posterior = (np.eye(2) - l_t @ Hx) @ prior

            # --- Plot passo passo (solo posa reale e stimata) ---
            plt.figure(figsize=(6, 6))
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.grid(True)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'EKF step {t}')

            # Posa reale (verde) e stimata (rossa)
            plt.plot(true_position[0], true_position[1], 'sg', markersize=8, label='Posa reale')
            plt.plot(x_est[0], x_est[1], 'or', markersize=8, label='Stima EKF')

            # Beacon (blu)
            plt.plot(m1[0], m1[1], 'db', label='Beacon 1')
            plt.plot(m2[0], m2[1], 'db', label='Beacon 2')
            plt.plot(m3[0], m3[1], 'db', label='Beacon 3')

            plt.legend()
            plt.show()

import numpy as np
import matplotlib.pyplot as plt
from filters import  *
from matplotlib.patches import Ellipse


USE_PARTICLE_FILTER = False
USE_KALMAN_FILTER = True

#Plot della covarianza della stima del filrto di Kalman
def covarianza_plot_ekf(mean, cov, n_std=2.0, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    plt.gca().add_patch(ellipse)


if __name__ == "__main__":

    # Posizione reale del robot e beacons
    true_position = np.array([0, 20], dtype=float)
    beacons = np.array([[8, 8],[7,2],[15,1]], dtype=float)

    #Parametri generali simulazione e parametri di rumore
    n_cicli = 20
    v_t_nominale = 1
    alpha_v = 0.01
    alpha_theta = 0.01
    sigma_z = 0.2

    # Mappa quadrata
    dim_mappa = 40

    if USE_PARTICLE_FILTER:
        n_particles = 3000


        #Scelgo x e y per ogni particella seguendo una distribuzione casuale
        particles_array_x = np.random.uniform(0,dim_mappa,n_particles)
        particles_array_y = np.random.uniform(0,dim_mappa,n_particles)

        #Assegno pesi uniformi a tutte le particelle(il robot potrebbe essere ovunque)
        particles_weight = np.ones(n_particles)/n_particles

        #Print dello stato iniziale
        plt.figure(figsize=(8, 8))
        plt.scatter(particles_array_x, particles_array_y, color="red")
        for i in range(len(beacons)):
            plt.plot(beacons[i][0], beacons[i][1], marker="s", color="green", label=f"beacon{i}")
        idx = np.argmax(particles_weight)
        x_stima = particles_array_x[idx]
        y_stima = particles_array_y[idx]
        plt.plot(x_stima, y_stima, marker='o', color='black', label='Stima')
        plt.scatter(particles_array_x, particles_array_y, color="red")
        plt.plot(true_position[0], true_position[1], marker="s", color="blue", label='True position')
        plt.legend()
        plt.show()


        #Cicllo for di simulazione temporale
        for t in range(n_cicli):

            #richiamo codice del particle filter
            true_position, v_t, x_stima, y_stima, particles_array_x, particles_array_y, particles_weight = particle_filter(
                t, true_position, v_t_nominale, alpha_v, alpha_theta, sigma_z,
                beacons, particles_array_x, particles_array_y, particles_weight
            )
            #Aggiorno la v per il passo successivo
            v_t_nominale = v_t

            plt.plot(x_stima, y_stima, marker='o', color='black', label='Stima')
            plt.scatter(particles_array_x,particles_array_y,color = "red")
            plt.plot(true_position[0], true_position[1], marker = "s", color="blue", label='True position')

            # Cerchi di distanza dai beacon rispetto alla posizione reale
            for i in range(len(beacons)):
                distanza = np.linalg.norm(true_position - beacons[i])
                cerchio = plt.Circle((beacons[i][0], beacons[i][1]), distanza, color='blue', fill=False,
                                     linestyle='--', label=f"Distanza trueposition-beacon{i}")
                plt.gca().add_patch(cerchio)

            #Gestione dinamica dei print dei beacon
            for i in range(len(beacons)):
                plt.plot(beacons[i][0], beacons[i][1], marker="s", color="green", label = f"beacon{i}")
            plt.legend()
            plt.title(f"ParticlesFilter - Ciclo {t}/{n_cicli}")
            plt.xlim(-10, dim_mappa)
            plt.ylim(-10, dim_mappa)
            plt.xlabel('Dimensione x')
            plt.ylabel('Dimensione y')
            plt.show()

    if USE_KALMAN_FILTER:
        #Inizializzazione della stima predetta
        x_stima = np.array([0,0])

        #Inizializzazione posterior e prior al passo zero
        var_x,var_y  = 10,10
        prior = np.diag([var_x,var_y])
        posterior = prior.copy()
        error_list = []
        #Inizializzo le traiettorie da plottare alla fine del ciclo temporale
        true_traj = [true_position.copy()]
        stima_traj = [x_stima.copy()]

        #Plot dello stato iniziale
        plt.figure()
        plt.plot(x_stima[0], x_stima[1], 'or', label='Stima EKF')
        plt.plot(true_position[0], true_position[1], 'sb', label='Posizione reale')
        for i in range(len(beacons)):
            plt.plot(beacons[i][0], beacons[i][1], 'sg', label=f"Beacon {i}")
        plt.legend()
        plt.xlim(-10, dim_mappa)
        plt.ylim(-10, dim_mappa)
        plt.title("Stato iniziale")
        plt.show()

        #Inizio ciclo temporale di simulazione
        for t in range(n_cicli):

            #Richiamo il codice del filtro di Kalman
            true_position, x_stima, v_t_nominale,prior,posterior, errore = kalman_filter(
                t, true_position, x_stima, v_t_nominale, alpha_v, alpha_theta,
                sigma_z, beacons, prior, posterior
            )

            #Aggiorno le traiettorie e l'andamento dell'errore
            error_list.append(errore)
            true_traj.append(true_position.copy())
            stima_traj.append(x_stima.copy())

            # Posa reale (verde) e stimata (rossa)
            plt.plot(x_stima[0], x_stima[1], 'or', markersize=8, label='Stima EKF')
            covarianza_plot_ekf(x_stima, posterior,n_std=3.0, edgecolor='red', linestyle='--', alpha=0.5)
            plt.plot(true_position[0], true_position[1], marker="s", color="blue", label='True position')
            # Gestione dinamica dei print dei beacon
            for i in range(len(beacons)):
                plt.plot(beacons[i][0], beacons[i][1], marker="s", color="green", label=f"beacon{i}")

            print(f"Errore di stima: {np.linalg.norm(true_position - x_stima)}\n")
            plt.legend()
            plt.title(f"Filtro EKF - Ciclo {t}/{n_cicli}")
            plt.xlim(-10, dim_mappa)
            plt.ylim(-10, dim_mappa)
            plt.show()

        #Plot dell'andamento dell'errore di stima
        plt.figure()
        plt.plot(range(n_cicli), error_list, marker='o', color='red')
        plt.title("Errore di stima EKF nel tempo")
        plt.xlabel("Tempo (cicli)")
        plt.ylabel("Errore di stima")
        plt.grid(True)
        plt.show()


        #Plot delle traiettorie reali e stimate per verificarne le differenzen e la convergenza del filtro
        true_traj = np.array(true_traj)
        stima_traj = np.array(stima_traj)
        plt.figure()
        plt.plot(true_traj[:, 0], true_traj[:, 1], '-ob', label='Posizione reale')
        plt.plot(stima_traj[:, 0], stima_traj[:, 1], '-or', label='Stima EKF')
        for i in range(len(beacons)):
            plt.plot(beacons[i][0], beacons[i][1], 'sg', label=f"Beacon {i}")
        plt.title("Confronto traiettoria reale vs stimata")
        plt.xlabel("Dimensione x")
        plt.ylabel("Dimensione y")
        plt.xlim(-10, dim_mappa)
        plt.ylim(-10, dim_mappa)
        plt.legend()
        plt.grid(True)
        plt.show()
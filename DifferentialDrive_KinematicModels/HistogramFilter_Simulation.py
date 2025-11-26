import numpy as np
import matplotlib.pyplot as plt
from HistogramFilter_v2 import  motion_model,measurement_model

if __name__ == "__main__":
    l_map = 30
    dim_bins = 0.1
    n_bins = int(l_map / dim_bins)
    Map = np.linspace(0, l_map, n_bins)

    # Posizione beacons
    m1 = Map[15]
    m2 = Map[30]

    #Parametri movimento e misure
    ux = 10
    sigma_x = 0.8
    sigma_z = 0.01

    x_t_prec = Map[0]
    prior = np.ones(n_bins) / n_bins
    posterior=  np.ones(n_bins) / n_bins
    likelihood_1= np.ones(n_bins) / n_bins
    likelihood_2 = np.ones(n_bins) / n_bins


    for t in range(n_bins):
        #Ci muoviamo dulla mappa, calcoliamo xt e zt
        x_t_att = x_t_prec + ux + np.random.normal(0,sigma_x)
        z_t_1 = np.abs(x_t_att - m1) + np.random.normal(0,sigma_z)
        z_t_2 = np.abs(x_t_att - m2) + np.random.normal(0, sigma_z)

        for k in range(n_bins):
            prior_t_k = 0
            for i in range(n_bins):
                # attenzione: la posterior è quella precedente, ma devi valutarla nella xi che stai ciclando
                # così hai uno scalare che moltiplica uno scalare e ottieni pian piano una somma di scalari
                # che poi metti in prior[k]
                prior_t_k += motion_model(Map[k], Map[i], ux,sigma_x) * posterior[i]
            prior[k] = prior_t_k
            likelihood_1[k] = measurement_model(z_t_1, Map[k], m1,sigma_z)
            likelihood_2[k] = measurement_model(z_t_2, Map[k], m2,sigma_z)

        prior = prior / np.sum(prior)
        likelihood_1 = likelihood_1
        likelihood_2 = likelihood_2
        likelihood = likelihood_1 * likelihood_2
        likelihood = likelihood/np.sum(likelihood)
        posterior = likelihood * prior
        posterior = posterior /np.sum(posterior)
        x_t_prec = x_t_att

        #Stimo la posa prendendo l'indice del valore massimo nella posterior e usandolo per individuare il punto della mappa corrispondente
        stimaposa = Map[np.argmax(posterior)]
        x = np.linspace(0, l_map, n_bins)
        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        axs[0].plot(x, prior, 'blue')
        axs[0].plot(m1, 0, marker='D', color='orange', label='Beacon m1')
        axs[0].legend()
        axs[0].plot(m2, 0, marker='D', color='orange', label='Beacon m2')
        axs[0].legend()
        axs[0].plot(x_t_att, 0, marker='D', color='blue', label='Posa del robot reale')
        axs[0].legend()
        axs[0].set_title('Prior (predizione)')
        axs[1].plot(x, likelihood, 'r', label='Likelihood m1')
        axs[1].legend()
        axs[2].plot(x, posterior, 'green')
        axs[2].set_title('Posterior)')
        axs[2].plot(x_t_att, 0, marker='D', color='blue', label='Posa del robot reale')
        axs[2].plot(stimaposa, 0, marker='D', color='green', label='Posa Stimata')
        axs[2].legend()
        plt.show()
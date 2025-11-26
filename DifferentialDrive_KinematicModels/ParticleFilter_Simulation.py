import numpy as np
import matplotlib.pyplot as plt
from HistogramFilter_v2 import  measurement_model



if __name__ == "__main__":
    l_map = 30
    dim_bins = 0.5
    n_bins = int(l_map / dim_bins)

    # Posizione beacons
    m1 = 15
    #m2 = Map[30]

    # Parametri movimento e misure
    ux = 0.5
    sigma_x = 0.9
    sigma_z = 0.9

    #Generazione Particelle
    n_particles = 100
    particles_array = []
    #Genero casualmente N particelle , l'array contiene la coordinata sulla mappa e il peso(verosimiglianza rispetto alla lettura del sensore)
    #particles_array[pos][weight] per accederee ai vari valori
    weight_p = 1 / n_particles
    for i in range(n_particles):
        pos = np.random.uniform(0, l_map)
        particles_array.append([pos, weight_p])

    #Likelihood uniforme, pesi che do alle particelle
    likelihood= np.ones(n_bins) / n_bins
    x_t_prec = 0
    for t in range(n_bins):

        #Aggiornamento Movimento robot
        x_t_att = x_t_prec + ux + np.random.normal(0, sigma_x)
        z_t = np.abs(x_t_att - m1) + np.random.normal(0, sigma_z)

        weight_tot = 0
        for p in range(n_particles):
            #Muovo le particelle con il motion model e calcolo la likelihood per ognuna di esse
            x_t_parti = particles_array[p][0] + ux + np.random.normal(0, sigma_x)
            w = measurement_model(z_t, x_t_parti, m1,sigma_z)
            #Aggiorno posizione e peso delle particelle
            particles_array[p][0] = x_t_parti
            particles_array[p][1] = w
            #Accumulo il peso totale che mi serve per la normalizzazione
            weight_tot += w

        # Normalizzazione dei pesi secondo il peso totale
        if weight_tot == 0:
            # Se per qualche motivo tutti i pesi sono 0 → assegna pesi uniformi
            for p in range(n_particles):
                particles_array[p][1] = 1 / n_particles
        else:
            for p in range(n_particles):
                particles_array[p][1] /= weight_tot

        #array che conterrà le particelle resamplate
        particles_array_resampled = []

        #CDF della distribuzione delle particelle
        cumulative_sum = np.cumsum([w for _, w in particles_array])
        n_particles_to_sample = len(particles_array)
        #Resampling, prendo un numero a caso tra 0 e 1 , e con searchsorted prendo il valore della particella con quel peso
        for p in range(n_particles_to_sample):
            r = np.random.uniform(0, 1)
            # Utile quando ho una distribuzione crescente, mi dice a quale x corrisponde la y(in questo caso r)
            idx = np.searchsorted(cumulative_sum, r)
            pos = particles_array[idx][0]  # posizione della particella scelta
            particles_array_resampled.append([pos, 1 / n_particles])  # nuovo peso uniforme





        # --- Stima della posizione (moda o media pesata) ---
        positions = np.array([p[0] for p in particles_array])
        weights = np.array([p[1] for p in particles_array])

        # Istogramma pesato → moda approssimata
        hist, bin_edges = np.histogram(positions, bins=n_bins, range=(0, l_map), weights=weights)
        mode_idx = np.argmax(hist)
        x_est = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2




        #Plotto l'avanzamento del tutto perchè non so fare plot complicati :(
        x = np.linspace(0, l_map, n_bins)
        plt.figure(figsize=(9,9))
        partics_iniziale= np.array(particles_array)
        plt.scatter(partics_iniziale[:,0], np.zeros(len(partics_iniziale)), color='blue')
        plt.plot(m1, 0, marker='D', color='orange', label='Beacon m1')
        plt.plot(x_t_att, 0, marker='D', color='red', label='Posizione Reale')
        plt.plot(x_est, 0, marker='o', color='green', label='Stima (moda)')

        plt.legend()
        plt.xlim(0, l_map)
        plt.ylim(-0.5, 0.5)
        plt.title(f"Time step {t}")
        plt.show()

        particles_array = particles_array_resampled
        x_t_prec = x_t_att
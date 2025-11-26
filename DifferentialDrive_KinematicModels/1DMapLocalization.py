import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import  matplotlib.animation as animation

def histogram_step(Map, N, m1, m2, sigma, ux, bel, z1, z2):
    posterior_list = []
    for xk in range(len(Map)):
        #Ciclo sugli x_k
        prior = np.zeros(N)

        posterior_k = 0
        prior_k = 0
        for xi in range(len(Map)):
            #Calcolo della prior in base alla xk che sto prendendo in considerazione
            p_v = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((Map[xk] - (Map[xi] + ux))**2) / (2 * sigma**2))
            prior_k += p_v * bel[xi]

        prior[xk] = prior_k

        # Posterior, aggiustamento con la lettura zt e il beacon
        p_mu_1 = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-((z1 - abs(Map[xk] - m1)) ** 2) / (2 * sigma ** 2))
    # p_mu_2 = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-((z2 - abs(Map[xk] - m2)) ** 2) / (2 * sigma ** 2))

        posterior_k =p_mu_1 * prior_k
        posterior_list.append(posterior_k)

    return posterior_list

if __name__ == "__main__" :
    N = 100
    Map  = []
    dimBins = 0.1
    m1 = 4.2
    m2 = 6.0
    period = 100

    for i in range(N):
        Map = np.append(Map, dimBins * i)

    ux = 0.1

    # x_t = x_t_1 + ux + v
    sigma = 0.2
    mu = 0
    #Non so dove sta il robot, quindi ha probabilità uniforme di trovarsi in una delle celle
    bel = np.ones(N)/N
    x_true = 0.0
    x_true_list = []

    beliefs = []
    for t in range (period):
        x_true += ux + np.random.normal(0, sigma)  # movimento reale con rumore
        x_true_list.append(x_true)

        z1 = abs(x_true - m1) + np.random.normal(0, sigma)
        z2 = abs(x_true - m2) + np.random.normal(0, sigma)

        posterior_list = histogram_step(Map, N, m1, m2, sigma, ux, bel,z1,z2)
        posterior_array = np.array(posterior_list)
        posterior_array /= np.sum(posterior_array)  # normalizzazione
        bel = posterior_array  # aggiorna la belief per il prossimo passo
        beliefs.append(posterior_array)

    fig, ax = plt.subplots()
    ax.set_xlim(Map[0], Map[-1])
    ax.set_ylim(0, 0.40)
    ax.set_xlabel("Posizione x")
    ax.set_ylabel("Belief")
    ax.set_title("Evoluzione della belief nel tempo")

    robot_path, = ax.plot([], [], 'r-', lw=2, label='Belief')
    ax.legend()


    def animate(i):
        robot_path.set_data(Map, beliefs[i])
        return robot_path,


    ani = animation.FuncAnimation(fig, animate, frames=period, blit=True, interval=100)
    plt.show()

    posterior_array = np.array(posterior_list)
    posterior_array = posterior_array / np.sum(posterior_array)
    print(np.sum(posterior_array))

    plt.plot( Map,posterior_array, label="posterior")
    plt.legend()
    plt.title("Likelihoods per ciascun beacon")
    plt.xlabel("Posizione ipotetica xk")
    plt.ylabel("p(zt | xk, m)")
    plt.grid()
    plt.show()


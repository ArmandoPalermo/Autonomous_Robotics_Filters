import numpy as np
import matplotlib.pyplot as plt

def motion_model(x_att, x_pre, ux, sigma_x):
    return (1 / np.sqrt(2 * np.pi * sigma_x**2)) * np.exp(-0.5 * ((x_att - (x_pre + ux))**2) / sigma_x**2)


def measurement_model(z_att, x_att, m1, sigma_z):
    expected_z = np.abs(x_att - m1)
    return (1 / np.sqrt(2 * np.pi * sigma_z**2)) * np.exp(-0.5 * ((z_att - expected_z)**2) / sigma_z**2)



if __name__ == '__main__':
    l_map = 30
    dim_bins = 0.5
    n_bins = int(l_map / dim_bins)
    Map = np.linspace(0, l_map, n_bins)

    #Posizione beacon m1
    m1 = Map[10]
    m2 = Map[18]

    z_att = 2
    ux = 0.1
    sigma_x = 0.1
    sigma_z = 0.1
    eta = 0.5

    prior = np.ones(n_bins)/n_bins
    posterior = np.ones(n_bins)/n_bins
    posterior_prec = posterior
    likelihood_1 = np.ones(n_bins) / n_bins
    likelihood_2 = np.ones(n_bins) / n_bins
    print(n_bins)
    for k in range(n_bins):
        prior_t_k = 0
        for i in range(n_bins):
            #attenzione: la posterior è quella precedente, ma devi valutarla nella xi che stai ciclando
            #così hai uno scalare che moltiplica uno scalare e ottieni pian piano una somma di scalari
            #che poi metti in prior[k]
            prior_t_k += motion_model(Map[k],Map[i],ux, sigma_x) * posterior_prec[i]
        prior[k] = prior_t_k
        likelihood_1[k] = measurement_model(z_att, Map[k], m1,sigma_z)
        likelihood_2[k] = measurement_model(z_att, Map[k], m2,sigma_z)

    likelihood_1 = likelihood_1 / np.sum(likelihood_1)
    likelihood_2 = likelihood_2 / np.sum(likelihood_2)
    prior = prior / np.sum(prior)
    posterior_prec = eta * (likelihood_1* likelihood_2)  * prior


    x = np.linspace(0,l_map,n_bins)
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    axs[0].plot(x, prior, 'blue')
    axs[0].plot(m1, 0, marker='D', color='orange', label='Beacon m1')
    axs[0].legend()
    axs[0].plot(m2, 0, marker='D', color='orange', label='Beacon m2')
    axs[0].legend()
    axs[0].set_title('Prior (predizione)')
    axs[1].plot(x, likelihood_1, 'r', label='Likelihood m1')
    axs[1].plot(x, likelihood_2, 'm', label='Likelihood m2')
    axs[1].legend()
    axs[2].plot(x, posterior_prec,'green')
    axs[2].set_title('Posterior)')
    # Imposta i limiti degli assi (0 → l_map)
    for ax in axs:
        ax.set_xlim(0, l_map)

    plt.tight_layout()
    plt.show()



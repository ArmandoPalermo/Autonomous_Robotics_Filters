import numpy as np

def motion_model(t,true_position_x,true_position_y,v_t_nominale,alpha_v,alpha_theta):
    #Traiettoria sinusoidale ---> Attenzione, la velocita a volte puo diventare negativa (v_t_nominale parte da 1), e per gestire ciò uso un abs per tenerla positiva
    #Il max viene usato per evitare che sia troppo piccola o nulla
    v_t = v_t_nominale + np.cos(t)
    v_t = max(np.abs(v_t), 0.1)
    theta_t = np.pi / 4 * np.sin(0.5 * t)

    #Aggiunta di rumore dipendente da v
    v_t += np.random.normal(0,np.sqrt(alpha_v) * np.abs(v_t))
    theta_t += np.random.normal(0, np.sqrt(alpha_theta) * np.abs(v_t))

    true_position_x = true_position_x + v_t * np.cos(theta_t)
    true_position_y = true_position_y + v_t * np.sin(theta_t)

    return true_position_x,true_position_y,v_t,theta_t


#Measuerement model che misura la distanza tra la posizione attuale del robot/particella risetto al beacon
#Restituisce la verosimiglianza della misura e anche la distanza attesa
def measurement_model(z_att, x_att, m1, sigma_z):
    expected_z = np.linalg.norm(x_att - m1)
    likelihood = (1 / np.sqrt(2 * np.pi * sigma_z ** 2)) * np.exp(-0.5 * ((z_att - expected_z) ** 2) / sigma_z ** 2)
    likelihood = max(likelihood, 1e-8)
    return expected_z, likelihood
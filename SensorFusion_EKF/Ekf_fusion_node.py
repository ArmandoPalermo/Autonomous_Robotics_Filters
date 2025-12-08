import numpy as np


def quaternion_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw_rad = np.arctan2(siny_cosp, cosy_cosp)
    return yaw_rad  # Converti in radianti


def get_Q_Matrix(u):
    # Parametri scelti per la varianza
    alpha1 = 0.1
    alpha2 = 0.1
    alpha3 = 0.1
    alpha4 = 0.1


    var_v = alpha1 * (u[0] ** 2) + alpha2 * (np.abs(u[1]) ** 2)
    var_w = alpha3 * (u[0] ** 2) + alpha4 * (u[1] ** 2)

    Q = np.array([
        [var_v, 0],
        [0, var_w]
    ])
    return Q


def prediction_phase(x, P, last_u, delta_t):
    # FASE DI PREDIZIONE
    x_prev = x.copy()
    x_stima = motion_model(x_prev, last_u, delta_t)

    fx = jacobian_fx(x_prev, last_u, delta_t)
    fu = jacobian_fu(x_prev, last_u, delta_t)
    Q = get_Q_Matrix(last_u)

    P = fx @ P @ fx.T + fu @ Q @ fu.T

    return x_stima, P


# u = [v, omega]
def motion_model(x_prev, u, dt):
    v, w = u
    th = x_prev[2]

    x_new = x_prev.copy()
    x_new[0] += v * np.cos(th) * dt
    x_new[1] += v * np.sin(th) * dt
    x_new[2] += w * dt

    return x_new


def jacobian_fx(x, u, dt):
    v, _ = u
    theta = x[2]

    return np.array([
        [1., 0., -dt * v * np.sin(theta)],
        [0., 1.,  dt * v * np.cos(theta)],
        [0., 0., 1.]
    ])


def jacobian_fu(x, u, dt):
    theta = x[2]

    return np.array([
        [dt * np.cos(theta), 0.],
        [dt * np.sin(theta), 0.],
        [0.,                 dt]
    ])


def ekf_sensor_fusion(timeline, sensors):

    pose_stimate = []
    covariances = []
    inizialized = False

    last_u = np.array([0.1, 0.1])
    last_prediction_topic = ""
    last_correction_topic = ""
    n_same_topic = 0
    alpha  = 1
    alpha_veloce = 1
    for row in timeline:

        actual_sensor = sensors.get(row["topic"])

        # Utile in fase di test (per togliere sensori dal test)
        if actual_sensor is None:
            continue

        # Inizializzazione EKF al primo messaggio
        if not inizialized:

            x = row['msg'].pose.pose.position.x
            y = row['msg'].pose.pose.position.y
            qx = row['msg'].pose.pose.orientation.x
            qy = row['msg'].pose.pose.orientation.y
            qz = row['msg'].pose.pose.orientation.z
            qw = row['msg'].pose.pose.orientation.w

            theta = quaternion_to_yaw(qx, qy, qz, qw)

            # Stato iniziale [x, y, theta]
            x_stima = np.array([x, y, theta])

            # Covarianza iniziale
            P = np.diag([0.1, 0.1, (10 * np.pi / 180) ** 2])
            covariances.append(P.copy())

            # Salva la prima posa
            pose_stimate.append([row["time"], x, y, theta])

            inizialized = True
            last_time = row["time"]

            #Inizializzazione conteggio presenza topic(utile per scalare le R in base alla Freq)
            n_same_topic += 1
            if actual_sensor.has_correction_phase:
                last_correction_topic = actual_sensor.topic
        else:

            delta_t = row["time"] - last_time


            if actual_sensor.has_update_u_phase():
                last_u = np.array(actual_sensor.get_u_parameter(row))

            # Predizione solo quando dt > 0
            if delta_t > 0:
                x_stima, P = prediction_phase(x_stima, P, last_u, delta_t)
                last_prediction_topic = actual_sensor.topic
                last_time = row["time"]

            # Timestamp uguale → niente predizione doppia
            elif actual_sensor.has_update_u_phase():
                if actual_sensor.topic == last_prediction_topic:
                    last_u = actual_sensor.get_u_parameter(row)
                    continue

            # Fase di correzione
            if actual_sensor.has_correction_phase():


                if actual_sensor.topic == last_correction_topic:
                    n_same_topic +=1
                else:
                    print(n_same_topic)
                    # calcolo alpha in base a quanto era il rate per distinguere tra topic veloce e lento
                    #Se n_same_topic è minore di 1 allora vuol dire che il passaggio è stato lento --> veloce
                    #Se invece è maggiore allora veloce-->lento
                    if n_same_topic <= 1:
                        alpha = 10 * alpha_veloce
                    else:
                        alpha = 1 / n_same_topic
                        alpha = 0.1 * alpha
                        alpha_veloce = n_same_topic

                    n_same_topic = 1
                    last_correction_topic = actual_sensor.topic

                H = actual_sensor.get_H_matrix()
                R = alpha * actual_sensor.get_R_matrix(row)
                z = actual_sensor.get_z_measurements(row)
                print('MSG:',actual_sensor.topic,' R:',R)
                innovazione = z - H @ x_stima

                # Normalizza yaw (solo se presente nel vettore)
                if innovazione.shape[0] == 3:
                    innovazione[2] = (innovazione[2] + np.pi) % (2 * np.pi) - np.pi

                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x_stima = x_stima + K @ innovazione

                P = (np.eye(3) - K @ H) @ P
                # Normalizza theta
                x_stima[2] = (x_stima[2] + np.pi) % (2 * np.pi) - np.pi

            pose_stimate.append([row["time"], x_stima[0], x_stima[1], x_stima[2]])
            covariances.append(P.copy())
    return pose_stimate, covariances

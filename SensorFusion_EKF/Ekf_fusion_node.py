import numpy as np

def quaternion_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw_rad = np.arctan2(siny_cosp, cosy_cosp)
    return yaw_rad # Converti in radianti

def get_Q_Matrix(u):
    #Parametri da scgliere per la varianza
    alpha1 = 0.1
    alpha2 = 0.1
    alpha3 = 0.1
    alpha4 = 0.1

    var_v = alpha1 * (u[0] ** 2) + alpha2 * (u[1] ** 2)
    var_w = alpha3 * (u[0] ** 2) + alpha4 * (u[1] ** 2)
    Q = np.array([[var_v, 0],
                  [0, var_w]])
    return Q



def prediction_phase(x,P, last_u, delta_t):
    # FASE DI PREDIZIONE
    x_prev = x.copy()
    x_stima = motion_model(x_prev, last_u, delta_t)

    fx = jacobian_fx(x_prev, last_u, delta_t)
    fu = jacobian_fu(x_prev, last_u, delta_t)
    Q = get_Q_Matrix(last_u)

    P = fx @ P @ fx.T + fu @ Q @ fu.T

    return x_stima,P

#funzione motion model
# u = [v,omega]
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


#A questa funzione passi i dati estratti dal topic
def ekf_sensor_fusion(timeline,sensors):


    pose_stimate = []
    covariances = []
    inizialized = False

    last_u = np.array([0.1, 0.1])

    for row in timeline:
        actual_sensor = sensors.get(row["topic"])
        # Inizializzazione EKF al primo messaggio della rosbag
        if not inizialized:

            # Estrai posizione e orientamento
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
            P = np.diag([1.0, 1.0, (10 * np.pi / 180) ** 2])
            covariances.append(P.copy())

            # Salvo la prima posa
            pose_stimate.append([row["time"], x, y, theta])

            # Finito inizializzare
            inizialized = True
            last_time = row["time"]
        else:


            delta_t = row['time'] - last_time

            # IGNORA gli step con stesso timestamp
            if delta_t < 0:
                continue
            last_time = row['time']

            #Test solo ruote
            if actual_sensor is None:
                # fai SOLO predizione
                x_stima, P = prediction_phase(x_stima, P, last_u, delta_t)
                pose_stimate.append([row["time"], x_stima[0], x_stima[1], x_stima[2]])
                covariances.append(P.copy())
                continue

            if actual_sensor.has_update_u_phase():
                last_u = actual_sensor.get_u_parameter(row)

            x_stima,P = prediction_phase(x_stima,P,last_u, delta_t)

            if actual_sensor.has_correction_phase():

                H = actual_sensor.get_H_matrix()
                R = actual_sensor.get_R_matrix(row)
                z = actual_sensor.get_z_measurements(row)

                innovazione = z - H @ x_stima
                # Normalize yaw per tenerlo sempre tra -pi e pi(solo se presente nel vettore -->Dlio)
                if innovazione.shape[0] == 3:
                    innovazione[2] = (innovazione[2] + np.pi) % (2 * np.pi) - np.pi

                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x_stima = x_stima + K @ innovazione
                P = (np.eye(3) - K @ H) @ P

                # Normalize theta
                x_stima[2] = (x_stima[2] + np.pi) % (2 * np.pi) - np.pi

            pose_stimate.append([row["time"], x_stima[0], x_stima[1], x_stima[2]])
            covariances.append(P.copy())

    return pose_stimate, covariances

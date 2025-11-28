import numpy as np


def quaternion_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw_rad = np.arctan2(siny_cosp, cosy_cosp)
    return yaw_rad # Converti in radianti


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

# H dei tre sensori
#H_Dlio --> matrice diagonale 3x3 perchè effettuo una correzione sulle 3 comp dello stato
H_dlio = np.eye(3)
#H_wheel --> le ruote idelmente correggono anche x e y ma a noi interesa solo theta--> in quanto le velocità rendono piu sicura la sua stima, mentre il drift puo dar
#problemi durante la correzione(non ha peso)
H_wheel = np.array([[0., 0., 1.]])
#H_gps
H_gps = np.array([[1, 0, 0],
                  [0, 1, 0]])


#A questa funzione passi i dati estratti dal topic
def ekf_sensor_fusion(timeline):
    pose_stimate = []
    covariances = []
    inizialized = False

    last_u = np.array([0.0, 0.0])
    last_warthog_Q = np.diag([0.1 ** 2, (5 * np.pi / 180) ** 2])  # Q iniziale sicura finchè non arriva quella delle ruote
    #last_warthog_Q = np.eye(2)
    for row in timeline:

        if not inizialized:

            x = row['msg'].pose.pose.position.x
            y = row['msg'].pose.pose.position.y
            qx = row['msg'].pose.pose.orientation.x
            qy = row['msg'].pose.pose.orientation.y
            qz = row['msg'].pose.pose.orientation.z
            qw = row['msg'].pose.pose.orientation.w
            theta = quaternion_to_yaw(qx, qy, qz, qw)

            #Stima iniziale presa dal primo topic che arriva e covarianza alta
            x_stima = np.array([x, y, theta])
            P = np.diag([1.0, 1.0, (10 * np.pi / 180)**2])
            covariances.append(P.copy())
            #Salvo la prima posa
            pose_stimate.append([row["time"], x, y, theta])
            #Inizializzazione finita
            inizialized = True
            last_time = row["time"]

        else:

            #Calcolo il dt e procedo con la fase di PREDIZIONE
            delta_t = row['time'] - last_time
            last_time = row['time']

            x_prev = x_stima.copy()
            x_stima = motion_model(x_prev, last_u, delta_t)
            fx = jacobian_fx(x_prev, last_u, delta_t)
            fu = jacobian_fu(x_prev, last_u, delta_t)

            Q = last_warthog_Q
            P = fx @ P @ fx.T + fu @ Q @ fu.T

            if row['topic'] == 'dlio':
                #FASE DI PREDIZIONE
                cov = np.array(row['msg'].pose.covariance).reshape((6, 6))
                R_dlio = np.array([
                    [cov[0, 0], cov[0, 1], cov[0, 5]],
                    [cov[1, 0], cov[1, 1], cov[1, 5]],
                    [cov[5, 0], cov[5, 1], cov[5, 5]],
                ])

                z = np.array([
                    row['msg'].pose.pose.position.x,
                    row['msg'].pose.pose.position.y,
                    quaternion_to_yaw(
                        row['msg'].pose.pose.orientation.x,
                        row['msg'].pose.pose.orientation.y,
                        row['msg'].pose.pose.orientation.z,
                        row['msg'].pose.pose.orientation.w
                    )
                ])

                innovazione = z - H_dlio @ x_stima
                innovazione[2] = (innovazione[2] + np.pi) % (2 * np.pi) - np.pi

                S = H_dlio @ P @ H_dlio.T + R_dlio
                K = P @ H_dlio.T @ np.linalg.inv(S)
                x_stima = x_stima + K @ innovazione
                P = (np.eye(3) - K @ H_dlio) @ P

                x_stima[2] = (x_stima[2] + np.pi) % (2 * np.pi) - np.pi


            if row['topic'] == 'warthog':
                #Aggiornamento Q e ingressi
                last_u = np.array([
                    row['msg'].twist.twist.linear.x,
                    row['msg'].twist.twist.angular.z
                ])

                cov_twist = np.array(row['msg'].twist.covariance).reshape((6, 6))
                Q_warthog = np.array([
                    [cov_twist[0, 0], 0.],
                    [0.,             cov_twist[5, 5]]
                ])

                if np.linalg.det(Q_warthog) < 1e-12:
                    Q_warthog += np.eye(2) * 1e-3

                last_warthog_Q = Q_warthog

            if row['topic'] == 'gps':
                #Correzione GPS
                cov = np.array(row['msg'].pose.covariance).reshape((6, 6))
                R_gps = np.array([
                    [cov[0, 0], cov[0, 1]],
                    [cov[1, 0], cov[1, 1]]
                ])

                if np.linalg.det(R_gps) < 1e-12:
                    R_gps += np.eye(2) * 0.5

                z = np.array([
                    row['msg'].pose.pose.position.x,
                    row['msg'].pose.pose.position.y
                ])

                z_pred = H_gps @ x_stima
                innovazione = z - z_pred

                S = H_gps @ P @ H_gps.T + R_gps
                K = P @ H_gps.T @ np.linalg.inv(S)

                x_stima = x_stima + K @ innovazione
                P = (np.eye(3) - K @ H_gps) @ P

                x_stima[2] = (x_stima[2] + np.pi) % (2 * np.pi) - np.pi
            pose_stimate.append([row["time"], x_stima[0], x_stima[1], x_stima[2]])
            covariances.append(P.copy())

    return pose_stimate, covariances

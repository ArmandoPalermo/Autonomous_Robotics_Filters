import numpy as np
from Ekf_fusion_node import quaternion_to_yaw


class GpsSensor:
    topic = "gps"

    # Metodo che definisce se un sensore viene usato in fase di correzione ho meno
    def has_correction_phase(self):
        return True

    # Metodo che definisce se un sensore viene usato per aggiornare il controllo u da dare al motion model
    def has_update_u_phase(self):
        return False

    # Metodo get per ottenere le misurazioni dal sensore
    def get_z_measurements(self, row):
        z = np.array([
            row['msg'].pose.pose.position.x,
            row['msg'].pose.pose.position.y
        ])
        return z

    # Metodo get per ottenere la matrice H usata per la fase di correzione
    def get_H_matrix(self):
        H_gps = np.array([[1, 0, 0],
                          [0, 1, 0]])
        return  H_gps

    # Metodo get per ottenere la matrice R(rumore di misura)
    def get_R_matrix(self,row):
        scaleFactor = 1
        cov = np.array(row['msg'].pose.covariance).reshape((6, 6))
        R_gps = np.array([
            [cov[0, 0], cov[0, 1]],
            [cov[1, 0], cov[1, 1]]
        ])

        #print(R_gps)
        return R_gps * scaleFactor

    # Metodo per ottenere i parametri di velocità dal topic
    def get_u_parameter(self,row):
        u = np.array([
           - row['msg'].twist.twist.linear.x,
            row['msg'].twist.twist.angular.z
        ])

        return u




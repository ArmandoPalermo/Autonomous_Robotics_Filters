import numpy as np

from Ekf_fusion_node import quaternion_to_yaw

class WarthogWheelSensor:

    topic = "warthog"

    # Metodo che definisce se un sensore viene usato in fase di correzione ho meno
    def has_correction_phase(self):
        return False

    # Metodo che definisce se un sensore viene usato per aggiornare il controllo u da dare al motion model
    def has_update_u_phase(self):
        return True

    # Metodo get per ottenere le misurazioni dal sensore
    def get_z_measurements(self,row):
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
        return z

    # Metodo get per ottenere la matrice H usata per la fase di correzione
    def get_H_matrix(self):
        #Correggi theta?
        return np.array([0,0,1])

    # Metodo get per ottenere la matrice R(rumore di misura)
    def get_R_matrix(self,row):
        scaleFactor = 1
        cov = np.array(row['msg'].pose.covariance).reshape((6, 6))
        R_wheel = np.array([
            [cov[0, 0], cov[0, 1], cov[0, 5]],
            [cov[1, 0], cov[1, 1], cov[1, 5]],
            [cov[5, 0], cov[5, 1], cov[5, 5]],
        ])
        return R_wheel * scaleFactor

    # Metodo per ottenere i parametri di velocità dal topic
    def get_u_parameter(self,row):
        u = np.array([
           - row['msg'].twist.twist.linear.x,
            row['msg'].twist.twist.angular.z
        ])

        return u




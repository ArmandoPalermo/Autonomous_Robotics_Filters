import numpy as np
from Ekf_fusion_node import quaternion_to_yaw

class DlioSensor:

    topic = "dlio"

<<<<<<< HEAD
    def has_correction_phase(self):
        return True

    def has_update_u_phase(self):
        return False

=======
    #Metodo che definisce se un sensore viene usato in fase di correzione ho meno
    def has_correction_phase(self):
        return True

    #Metodo che definisce se un sensore viene usato per aggiornare il controllo u da dare al motion model
    def has_update_u_phase(self):
        return False

    #Metodo get per ottenere le misurazioni dal sensore
>>>>>>> fa9b12db0f5fd97ca6665628c95ce1e2b1134118
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

<<<<<<< HEAD
    def get_H_matrix(self):
        return np.eye(3)

    def get_R_matrix(self, row):
        cov = np.array(row["msg"].pose.covariance).reshape(6, 6)

        # estrai i tre elementi che ti interessano
=======
    #Metodo get per ottenere la matrice H usata per la fase di correzione
    def get_H_matrix(self):
        return np.eye(3)

    #Metodo get per ottenere la matrice R(rumore di misura)
    def get_R_matrix(self, row):
        cov = np.array(row["msg"].pose.covariance).reshape(6, 6)

        #Estrai gli elementi della diagonale corrispondenti a x,y,theta
>>>>>>> fa9b12db0f5fd97ca6665628c95ce1e2b1134118
        var_x = max(cov[0, 0], 1e-6)
        var_y = max(cov[1, 1], 1e-6)
        var_yaw = max(cov[5, 5], 1e-6)

<<<<<<< HEAD
        # scaling separato
        scale_xy = 1 # come il tuo attuale scaleFactor per x,y
        scale_yaw = 1 # yaw meno scalato perché più attendibile
=======
        #Scaling applicato a dlio in maniera separata
        scale_xy = 2 #Fattore di scala applicato a x,y
        scale_yaw = 1 #Fattore di scala applicato a theta
>>>>>>> fa9b12db0f5fd97ca6665628c95ce1e2b1134118


        R_dlio = np.array([
            [var_x * scale_xy, 0.0, 0.0],
            [0.0, var_y * scale_xy, 0.0],
            [0.0, 0.0, var_yaw * scale_yaw]
        ])

        #print(R_dlio)
        return R_dlio

<<<<<<< HEAD
=======
    #Metodo per ottenere i parametri di velocità dal topic
>>>>>>> fa9b12db0f5fd97ca6665628c95ce1e2b1134118
    def get_u_parameter(self,row):
        u = np.array([
            row['msg'].twist.twist.linear.x,
            row['msg'].twist.twist.angular.z
        ])

        return u

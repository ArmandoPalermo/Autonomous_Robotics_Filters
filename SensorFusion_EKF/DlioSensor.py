import numpy as np
from Ekf_fusion_node import quaternion_to_yaw

class DlioSensor:

    topic = "dlio"

    def has_correction_phase(self):
        return True

    def has_update_u_phase(self):
        return False

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

    def get_H_matrix(self):
        return np.eye(3)

    def get_R_matrix(self, row):
        cov = np.array(row["msg"].pose.covariance).reshape(6, 6)

        # estrai i tre elementi che ti interessano
        var_x = max(cov[0, 0], 1e-6)
        var_y = max(cov[1, 1], 1e-6)
        var_yaw = max(cov[5, 5], 1e-6)

        # scaling separato
        scale_xy = 1 # come il tuo attuale scaleFactor per x,y
        scale_yaw = 1 # yaw meno scalato perché più attendibile


        R_dlio = np.array([
            [var_x * scale_xy, 0.0, 0.0],
            [0.0, var_y * scale_xy, 0.0],
            [0.0, 0.0, var_yaw * scale_yaw]
        ])

        #print(R_dlio)
        return R_dlio

    def get_u_parameter(self,row):
        u = np.array([
            row['msg'].twist.twist.linear.x,
            row['msg'].twist.twist.angular.z
        ])

        return u

import numpy as np
from Ekf_fusion_node import quaternion_to_yaw


class GpsSensor:
    topic = "gps"

    def has_correction_phase(self):
        return True

    def has_update_u_phase(self):
        return False

    def get_z_measurements(self, row):
        z = np.array([
            row['msg'].pose.pose.position.x,
            row['msg'].pose.pose.position.y
        ])
        return z

    def get_H_matrix(self):
        H_gps = np.array([[1, 0, 0],
                          [0, 1, 0]])
        return  H_gps

    def get_R_matrix(self,row):
        scaleFactor = 1
        cov = np.array(row['msg'].pose.covariance).reshape((6, 6))
        R_gps = np.array([
            [cov[0, 0], cov[0, 1]],
            [cov[1, 0], cov[1, 1]]
        ])

        #print(R_gps)
        return R_gps * scaleFactor




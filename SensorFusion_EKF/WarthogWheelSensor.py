
import numpy as np
from Ekf_fusion_node import quaternion_to_yaw

class WarthogWheelSensor:

    topic = "warthog"

    def has_correction_phase(self):
        return False

    def has_update_u_phase(self):
        return True

    def get_u_parameter(self,row):
        u = np.array([
            row['msg'].twist.twist.linear.x,
            row['msg'].twist.twist.angular.z
        ])

        return u




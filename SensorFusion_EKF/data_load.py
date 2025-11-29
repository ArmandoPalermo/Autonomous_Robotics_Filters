import rosbag
import numpy as np
import pandas as pd
from Ekf_fusion_node import quaternion_to_yaw


def load_data(bag_path):

    print(f"Caricamento dati da: {bag_path}")

    bag = rosbag.Bag(bag_path)

    # Struttura generale
    data = {
        'gps': {},
        'dlio': {},
        'warthog': {}
    }

    # Variabili per ogni topic
    for key in data.keys():
        data[key] = {
            'time': [],
            'x': [], 'y': [], 'z': [],
            'qx': [], 'qy': [], 'qz': [], 'qw': [],
            'cov': [],
            # campi twist -> saranno riempiti solo quando esistono
            'vx': [], 'vy': [], 'vz': [],
            'wx': [], 'wy': [], 'wz': []
        }

    topic_map = {
        '/odometry/gps': 'gps',
        '/robot/dlio/odom_node/odom': 'dlio',
        '/warthog_velocity_controller/odom': 'warthog'
    }

    timeline = []

    #Lettura della rosbag messaggio per messaggio
    for topic, msg, t in bag.read_messages():

        if topic not in topic_map:
            continue

        key = topic_map[topic]

        timestamp = t.to_sec()

        timeline.append({
            "time": timestamp,
            "topic": key,
            "msg": msg
        })

        #  POSA
        data[key]['time'].append(timestamp)
        data[key]['x'].append(msg.pose.pose.position.x)
        data[key]['y'].append(msg.pose.pose.position.y)
        data[key]['z'].append(msg.pose.pose.position.z)

        data[key]['qx'].append(msg.pose.pose.orientation.x)
        data[key]['qy'].append(msg.pose.pose.orientation.y)
        data[key]['qz'].append(msg.pose.pose.orientation.z)
        data[key]['qw'].append(msg.pose.pose.orientation.w)

        cov_matrix = np.array(msg.pose.covariance).reshape(6, 6)
        data[key]['cov'].append(cov_matrix)

        #Aggiunta dati twist(SE ESISTE--> gps le ha a 0)
        if hasattr(msg, "twist"):
            data[key]["vx"].append(msg.twist.twist.linear.x)
            data[key]["vy"].append(msg.twist.twist.linear.y)
            data[key]["vz"].append(msg.twist.twist.linear.z)

            data[key]["wx"].append(msg.twist.twist.angular.x)
            data[key]["wy"].append(msg.twist.twist.angular.y)
            data[key]["wz"].append(msg.twist.twist.angular.z)
        else:
            # Non esiste twist → metti NaN
            data[key]["vx"].append(np.nan)
            data[key]["vy"].append(np.nan)
            data[key]["vz"].append(np.nan)
            data[key]["wx"].append(np.nan)
            data[key]["wy"].append(np.nan)
            data[key]["wz"].append(np.nan)

    bag.close()

    # --- ORDINA TIMELINE ---
    timeline = sorted(timeline, key=lambda x: x["time"])

    # --- COSTRUZIONE DATAFRAME ---
    result = {}
    start_time = timeline[0]['time']

    for key in data.keys():

        result[key] = pd.DataFrame({
            'time': np.array(data[key]['time']) - start_time,
            'x': data[key]['x'],
            'y': data[key]['y'],
            'z': data[key]['z'],
            'qx': data[key]['qx'],
            'qy': data[key]['qy'],
            'qz': data[key]['qz'],
            'qw': data[key]['qw'],
            'vx': data[key]['vx'],
            'vy': data[key]['vy'],
            'vz': data[key]['vz'],
            'wx': data[key]['wx'],
            'wy': data[key]['wy'],
            'wz': data[key]['wz'],
            'cov_trace': [cov[0,0] + cov[1,1] + cov[5,5] for cov in data[key]['cov']]
        })

        # Calcolo yaw
        result[key]['theta'] = quaternion_to_yaw(
            result[key]['qx'].values,
            result[key]['qy'].values,
            result[key]['qz'].values,
            result[key]['qw'].values
        )

        print(f"{key}: {len(result[key])} samples (pose)")

        if key == "warthog":
            valid_twist = np.sum(~np.isnan(result[key]['vx']))
            print(f"     {valid_twist} samples con twist (v,w)")

    print("Dati caricati correttamente.\n")

    return result, timeline

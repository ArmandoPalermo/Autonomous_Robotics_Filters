import rosbag
import numpy as np
import pandas as pd
from Ekf_fusion_node import quaternion_to_yaw



def load_data(bag_path):

    print(f"Caricamento dati da: {bag_path}")

    bag = rosbag.Bag(bag_path)

    # Inizializza dizionari per salvare i dati
    data = {
        'gps': {'time': [], 'x': [], 'y': [], 'z': [], 'qx': [], 'qy': [], 'qz': [], 'qw': [], 'cov': []},
        'dlio': {'time': [], 'x': [], 'y': [], 'z': [], 'qx': [], 'qy': [], 'qz': [], 'qw': [], 'cov': []},
        'warthog': {'time': [], 'x': [], 'y': [], 'z': [], 'qx': [], 'qy': [], 'qz': [], 'qw': [], 'cov': []}
    }

    # Mappa i nomi dei topic alle chiavi del dizionario
    topic_map = {
        '/odometry/gps': 'gps',
        '/robot/dlio/odom_node/odom': 'dlio',
        '/warthog_velocity_controller/odom': 'warthog'
    }

    timeline = []
    # Leggi tutti i messaggi dal rosbag
    for topic, msg, t in bag.read_messages():
        if topic in topic_map:



            key = topic_map[topic]

            timeline.append({
                "time": t.to_sec(),
                "topic": key,  # gps / dlio / warthog
                "msg": msg
            })

            # Salva timestamp (usa il timestamp del rosbag, non del messaggio)
            data[key]['time'].append(t.to_sec())

            # Salva posizione (x, y, z)
            data[key]['x'].append(msg.pose.pose.position.x)
            data[key]['y'].append(msg.pose.pose.position.y)
            data[key]['z'].append(msg.pose.pose.position.z)

            # Salva orientamento (quaternion)
            data[key]['qx'].append(msg.pose.pose.orientation.x)
            data[key]['qy'].append(msg.pose.pose.orientation.y)
            data[key]['qz'].append(msg.pose.pose.orientation.z)
            data[key]['qw'].append(msg.pose.pose.orientation.w)

            # Salva matrice covarianza 6x6
            cov_matrix = np.array(msg.pose.covariance).reshape(6, 6)
            data[key]['cov'].append(cov_matrix)

    bag.close()

    #ordino la timeline
    timeline = sorted(timeline, key=lambda x: x["time"])

    # Converti in DataFrame per ogni topic
    result = {}
    for key in ['gps', 'dlio', 'warthog']:
        # Crea DataFrame con i dati base
        result[key] = pd.DataFrame({
            'time': data[key]['time'],
            'x': data[key]['x'],
            'y': data[key]['y'],
            'z': data[key]['z'],
            'qx': data[key]['qx'],
            'qy': data[key]['qy'],
            'qz': data[key]['qz'],
            'qw': data[key]['qw']
        })

        # Calcola traccia della covarianza per ogni sample
        result[key]['cov_trace'] = [np.trace(cov) for cov in data[key]['cov']]

        # Converti quaternion in angolo theta (yaw - rotazione attorno a Z)
        result[key]['theta'] = quaternion_to_yaw(
            result[key]['qx'].values,
            result[key]['qy'].values,
            result[key]['qz'].values,
            result[key]['qw'].values
        )

        # Normalizza il tempo a partire da zero
        result[key]['time'] = result[key]['time'] - result[key]['time'].min()

        print(f"  {key}: {len(result[key])} samples")

    print("Dati caricati!\n")
    return result, timeline

from data_plot import  *
from data_load import *
from pathlib import Path
from Ekf_fusion_node import ekf_sensor_fusion
from DlioSensor import DlioSensor
from GpsSensor import  GpsSensor
from WarthogWheelSensor import WarthogWheelSensor

# Nome del file rosbag nella cartella Rosbag/
BAG_FILENAME = "Rosbag_Warthog_PercorsoVerde_AR_2526-01-22_4_topics_map_drift.bag"


if __name__ == "__main__":
    # Path del rosbag
    bag_path = Path("Rosbag") / BAG_FILENAME

    # Carica i dati
    data, timeline = load_data(str(bag_path))
    #In timeline ci sono i dati aggiornati e allineati per tempo
    # Lettura dei diziona
    sensors = {
        "dlio": DlioSensor(),
        "gps": GpsSensor(),
        "warthog": WarthogWheelSensor()
    }

    pose_stima, posterior = ekf_sensor_fusion(timeline,sensors)
    start_time = timeline[0]["time"]


    # Converte pose_stima in un dict simile agli altri
    ekf_data = {
        "time": [p[0] - start_time for p in pose_stima],
        "x": [p[1] for p in pose_stima],
        "y": [p[2] for p in pose_stima],
        "theta": [p[3] for p in pose_stima]
    }


    plot_trajectory(data, ekf_data)
    plot_cov_trace(data,posterior,ekf_data["time"])
    plt.show()

    # Mostra tutti i plot
    plt.show()

    #print("Fatto!")
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(data, ekf_data):
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'green']
    labels = ['0. GPS', '1. DLIO', '2. Warthog Velocity']

    # Plots originali
    for key, color, label in zip(['gps', 'dlio', 'warthog'], colors, labels):
        plt.plot(data[key]['x'], data[key]['y'],
                 color=color, label=label, linewidth=2, alpha=0.8)

    # --- EKF fusion (traiettoria finale) ---
    plt.plot(ekf_data['x'], ekf_data['y'],
             color='black', linewidth=3, label='3. EKF Fusion')

    plt.xlabel('X [m]', fontsize=12, fontweight='bold')
    plt.ylabel('Y [m]', fontsize=12, fontweight='bold')
    plt.title('Traiettoria: Y vs X', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')


def plot_cov_trace(data, posterior):

    plt.figure(figsize=(12, 6))

    # --- Sensori ---
    for key, color in zip(["gps", "dlio", "warthog"],
                          ["blue", "red", "green"]):

        if key not in data:
            continue

        trace_values = data[key]["cov_trace"].values
        max_val = np.max(trace_values)

        if max_val == 0:
            continue

        trace_norm = trace_values / max_val  # NORMALIZZAZIONE

        plt.plot(
            data[key]["time"],
            trace_norm,
            label=f"{key.upper()} trace(P) (normalized)",
            linewidth=2,
            alpha=0.8,
            color=color
        )

    #  EKF trace
    ekf_trace = np.array([np.trace(P) for P in posterior])
    ekf_trace_norm = ekf_trace / np.max(ekf_trace)

    ekf_time = np.linspace(
        data["dlio"]["time"].min(),
        data["dlio"]["time"].max(),
        len(ekf_trace)
    )

    plt.plot(
        ekf_time,
        ekf_trace_norm,
        label="EKF trace(P) (normalized)",
        linewidth=3,
        color="black"
    )

    plt.title("Normalized Covariance Trace – Sensors vs EKF")
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized trace(P)  [0–1]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()

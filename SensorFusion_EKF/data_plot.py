
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




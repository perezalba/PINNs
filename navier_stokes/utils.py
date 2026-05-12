import numpy as np
import matplotlib.pyplot as plt

def get_poiseuille_velocity(y, u_max, h_max):
    return u_max * (1 - (y / h_max) ** 2)

def plot_flow_fields(samples, result):
    color_legend = [[0, 1.1], [-0.05, 0.05], [0, 0.6]]
    magnituds = [r"Velocity $u$ (m/s)", r"Velocity $v$ (m/s)", r"Pressure $p$ (Pa)"]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 9), sharex=True)

    xmin, xmax = samples[:, 0].min(), samples[:, 0].max()
    ymin, ymax = samples[:, 1].min(), samples[:, 1].max()

    for idx, ax in enumerate(axes):
        sc = ax.scatter(samples[:, 0], samples[:, 1], c=result[:, idx], cmap="jet", s=1)
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        ax.set_xticks(np.linspace(xmin, xmax, 6))
        ax.set_yticks(np.linspace(ymin, ymax, 5))
        
        ax.set_title(magnituds[idx])
        ax.set_ylabel("y")
        
        sc.set_clim(color_legend[idx])

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(magnituds[idx])

    axes[-1].set_xlabel("x")

    plt.tight_layout()
    plt.show()

def plot_velocity_profile(y_eval, u_pred, y_teoric, u_teoric, x_eval, H):
    plt.figure(figsize=(8, 6), dpi=150)

    plt.plot(u_teoric, y_teoric, 'k-', linewidth=2, label='Analytical Solution (Poiseuille)')
    plt.scatter(u_pred, y_eval, color='red', marker='o', s=15, alpha=0.7, label='PINN Prediction', zorder=5)

    plt.axhline(-H/2, color='gray', linestyle='--', linewidth=1)
    plt.axhline(H/2, color='gray', linestyle='--', linewidth=1)

    plt.title(f'Velocity $u$ at $x = {x_eval}$ m')
    plt.xlabel(r'Velocity $u$ (m/s)')
    plt.ylabel(r'Vertical position $y$ (m)')
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
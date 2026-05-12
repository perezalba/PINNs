import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_flow_fields_rectangle(samples, result, obs_coords):
    magnituds = [r"Velocity $u$ (m/s)", r"Velocity $v$ (m/s)", r"Pressure $p$ (Pa)"]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True, dpi=150)

    xmin, xmax = samples[:, 0].min(), samples[:, 0].max()
    ymin, ymax = samples[:, 1].min(), samples[:, 1].max()
    
    x_bottom_left, y_bottom_left, width, height = obs_coords

    for idx, ax in enumerate(axes):
        sc = ax.scatter(samples[:, 0], samples[:, 1], c=result[:, idx], cmap="jet", s=0.5)
        
        obstacle = Rectangle((x_bottom_left, y_bottom_left), width, height, 
                             color='dimgray', zorder=10)
        ax.add_patch(obstacle)
        
        ax.set_aspect('equal')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        ax.set_xticks(np.linspace(xmin, xmax, 6))
        ax.set_yticks(np.linspace(ymin, ymax, 5))
        
        ax.set_title(magnituds[idx])
        ax.set_ylabel("y")
        
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(magnituds[idx])

    axes[-1].set_xlabel("x")
    plt.tight_layout()    
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def tensor_heatmap(occupancy,velo):
    occupancy_channel = occupancy[0, :, :, 1]
    velo_channel = velo[0, :, :, 1, 1]
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Gráfico de ocupação
    cax1 = ax[0].imshow(occupancy_channel, cmap='viridis', interpolation='nearest')
    fig.colorbar(cax1, ax=ax[0], orientation='vertical', label='Occupancy')
    ax[0].set_title('Occupancy Grid')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    # Gráfico de velo
    cax2 = ax[1].imshow(velo_channel, cmap='plasma', interpolation='nearest')
    fig.colorbar(cax2, ax=ax[1], orientation='vertical', label='Velo')
    ax[1].set_title('Velo Grid')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')

    # Sincronizar os eixos
    for a in ax:
        a.set_xlim([0, 255])
        a.set_ylim([0, 255])
        a.set_xticks(np.arange(0, 256, 32))
        a.set_yticks(np.arange(0, 256, 32))

    plt.tight_layout()
    plt.show()
    exit()

def tensor_see(occupancy,velo,trust_treshould):
    if occupancy.ndim == 4:
        occupancy_channel = occupancy[0, :, :, :]
        vx = velo[0, :, :, :, 0]
        vy = velo[0, :, :, :, 1]
    else:
        occupancy_channel = occupancy[:, :, :]
        vx = velo[:, :, :, 0]
        vy = velo[:, :, :, 1]
    threshold = trust_treshould
    mask = occupancy_channel > threshold

    angles = np.arctan2(vy, vx)
    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = cm.get_cmap('hsv')
    colors = cmap(norm(angles))    

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plota um heatmap para visualização de ocupação
    occupancy_heatmap = np.max(occupancy_channel, axis=2)
    cax = ax.imshow(occupancy_heatmap, cmap='hot', interpolation='nearest')
    fig.colorbar(cax, ax=ax, orientation='vertical', label='Occupancy Score')
    
    # Plota as setas (quiver plot) para as velocidades
    for anchor in range(2):
        mask_anchor = mask[:, :, anchor]
        y_coords, x_coords = np.where(mask_anchor)
        ax.quiver(x_coords, y_coords, 
              vx[:, :, anchor][mask_anchor], vy[:, :, anchor][mask_anchor], 
              color=colors[:, :, anchor][mask_anchor], angles='xy', scale_units='xy',width= 0.005, scale=0.5)
        
    ax.set_title('Occupancy and Velocity Vectors')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, 255)
    ax.set_ylim(255, 0)
    ax.set_xticks(np.arange(0, 256, 32))
    ax.set_yticks(np.arange(0, 256, 32))
    plt.show()
    # exit()



if __name__ =='__main__':
    tensor_see()
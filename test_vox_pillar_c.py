import numpy as np
import config_model as cfg
import vox_pillar_l_c_plus  # O módulo C++ compilado
import time

# Parâmetros de configuração (certifique-se de que o módulo config_model está disponível)
image_size = cfg.image_shape
input_pillar_l_shape = cfg.input_pillar_l_shape
input_pillar_l_indices_shape = cfg.input_pillar_l_indices_shape
max_group = cfg.max_group_l
max_pillars = cfg.max_pillars_l

x_min = cfg.x_min
x_max = cfg.x_max
x_diff = cfg.x_diff

y_min = cfg.y_min
y_max = cfg.y_max
y_diff = cfg.y_diff

z_min = cfg.z_min
z_max = cfg.z_max
z_diff = cfg.z_diff

if __name__ == '__main__':
    start = time.time()
    lidar_path = 'C:/Users/maped/Documents/Scripts/Nuscenes/Lidar/'
    data = '0a0d1f7700da446580874d7d1e9fce51'
    lidar = np.load(lidar_path + data + '.npy')  # [0,1,2] -> Z,X,Y

    # Certifique-se de que os dados estão no tipo correto
    lidar = lidar.astype(np.float32)

    # Chamando a função do módulo C++
    vox_pillar_L, pos_L = vox_pillar_l_c_plus.pillaring_l(
        lidar,
        image_size,
        input_pillar_l_shape,
        input_pillar_l_indices_shape,
        max_group,
        max_pillars,
        x_min, x_diff,
        y_min, y_diff,
        z_min, z_diff
    )

    print("Vox Pillar L:", vox_pillar_L)
    print("Tempo de execução:", time.time() - start)
    print("Forma do vox_pillar_L:", vox_pillar_L.shape)
    print("Forma do pos_L:", pos_L.shape)

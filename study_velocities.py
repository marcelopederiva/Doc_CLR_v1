import matplotlib.pyplot as plt
import numpy as np

# Dados fornecidos
dados = [
    [-27.9, 25.2, -0.0, -0.0],
    [-16.3, 44.4, 1.941, 5.287],
    [-10.9, 12.6, -3.835, -4.433],
    [-10.9, 14.8, -3.581, -4.863],
    [-7.5, 24.0, -1.566, -5.011],
    [23.9, 36.4, 0.917, -1.397],
    [-0.7, 25.6, -0.109, -3.997],
    [-10.7, 64.4, 0.04, 0.243],
    [-10.3, 72.2, -0.0, -0.0]
]
dados_an = [
        [-27.34, 15.56, 0.06, -0.1],
    [-16.57, 47.26, 6.47, -7.71],
    [-11.3, 16.48, -6.86, -4.48],
    [-7.58, 26.86, -4.42, -5.1],
    [24.76, 34.31, 3.92, -5.27],
    [-0.21, 29.26, -1.73, -4.69],
    [-10.4, 68.96, 0.0, -0.0],
]
# Configuração do plano
x_min, x_max = -30, 30
y_min, y_max = 0, 60
grid_size = 256

# Criando a matriz de zeros 256x256x2
matriz = np.zeros((grid_size, grid_size, 2))

radar_t = np.load('radar_t.npy')
n_radar = np.zeros((radar_t.shape[0],4))
n_radar[:,0] = radar_t[:,0]
n_radar[:,1:] = radar_t[:,2:]
# print(n_radar)
# exit()
# Revisando a função de conversão de coordenadas para índices para lidar com casos fora dos limites
def coordenada_para_indice(x, y, x_min, x_max, y_min, y_max, grid_size):
    x_index = int(np.clip(((x - x_min) / (x_max - x_min)) * (grid_size - 1), 0, grid_size - 1))
    y_index = int(np.clip(((y - y_min) / (y_max - y_min) )* (grid_size - 1), 0, grid_size - 1))
    return y_index, x_index

# Recriando a matriz de zeros 256x256x2
matriz_rd = np.zeros((grid_size, grid_size, 2))

# Atualizando a matriz com os valores de velocidade
for x, y, vx, vy in n_radar:
    x_idx, y_idx = coordenada_para_indice(x, y, x_min, x_max, y_min, y_max, grid_size)
    # print(x,y)
    # print(x_idx,y_idx)
    # input()
    matriz_rd[x_idx, y_idx, 0] = (vx)
    matriz_rd[x_idx, y_idx, 1] = (vy)

# Plotando os vetores
# X = np.arange(0, grid_size, 1)
# Y = np.arange(0, grid_size, 1)
X, Y = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
U = matriz_rd[:, :, 0]
V = matriz_rd[:, :, 1]





# Recriando a matriz de zeros 256x256x2
# matriz_ann = np.zeros((grid_size, grid_size, 2))
# Atualizando a matriz com os valores de velocidade
# for x, y, vx, vy in dados_an:
#     x_idx, y_idx = coordenada_para_indice(x, y, x_min, x_max, y_min, y_max, grid_size)
#     matriz_ann[x_idx, y_idx, 0] = (vx)
#     matriz_ann[x_idx, y_idx, 1] = (vy)

tensor_ann = np.load('label_t.npy')
tensor_ann = tensor_ann[:,:,1,:]
tensor_ann = np.transpose(tensor_ann, axes=(1,0,2))
# print(tensor_ann.shape)
# exit()
X_an, Y_an = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
U_an = tensor_ann[:, :, 0]
V_an = tensor_ann[:, :, 1]




plt.figure(figsize=(10, 10))
plt.quiver(X_an, Y_an, U_an, V_an,scale =150, color='b')

plt.quiver(X, Y, U, V,scale =150, color='r')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
# plt.xlim(0, grid_size)
# plt.ylim(0, grid_size)
plt.legend(['Label','Radar'],loc='upper right')
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Velocity Vectors in each point')
# plt.grid()
# plt.savefig('RADARnLABEL.png')
plt.show()

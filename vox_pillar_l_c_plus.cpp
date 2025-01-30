#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <map>
namespace py = pybind11;
using namespace std;

typedef tuple<int, int> GridKey;

struct GridKeyHash {
    std::size_t operator()(const GridKey& key) const {
        return std::hash<int>()(std::get<0>(key)) ^ (std::hash<int>()(std::get<1>(key)) << 1);
    }
};

vector<py::array_t<float>> pillaring_l(py::array_t<float> cam_3d_input,
                                       py::tuple image_size,
                                       py::tuple input_pillar_l_shape,
                                       py::tuple input_pillar_l_indices_shape,
                                       int max_group,
                                       int max_pillars,
                                       float x_min, float x_diff,
                                       float y_min, float y_diff,
                                       float z_min, float z_diff) {
    // Acessando os dados de entrada
    auto cam_3d = cam_3d_input.mutable_unchecked<2>();

    
    pybind11::ssize_t num_points = cam_3d.shape(0);

    // Extraindo dimensões da imagem
    int image_width = image_size[0].cast<int>();
    int image_height = image_size[1].cast<int>();

    // Inicializando vetores
    vector<vector<float>> norm_i(num_points, vector<float>(4, 0.0f));
    vector<vector<float>> norm(num_points, vector<float>(2, 0.0f));
    vector<vector<float>> real_3d(num_points, vector<float>(3, 0.0f));
    // std::cout << "Before cam_3d(0, 1): " << cam_3d(0, 1) << std::endl;
    // Reordenando os eixos e normalizando os dados
    for (pybind11::ssize_t i = 0; i < num_points; ++i) {
        cam_3d(i, 1) = -cam_3d(i, 1); // Negando o eixo Y
    }
    // std::cout << "After cam_3d(0, 1): " << cam_3d(0, 1) << std::endl;
    for (pybind11::ssize_t i = 0; i < num_points; ++i) {
        // cam_3d(i, 1) = -cam_3d(i, 1); // Negando Y como no Python

        real_3d[i][0] = cam_3d(i, 1);
        real_3d[i][1] = cam_3d(i, 2);
        real_3d[i][2] = cam_3d(i, 0);

        norm_i[i][0] = norm[i][0] = (cam_3d(i, 1) - x_min) / x_diff;
        norm_i[i][1] = (cam_3d(i, 2) - y_min) / y_diff;
        norm_i[i][2] = norm[i][1] = (cam_3d(i, 0) - z_min) / z_diff;
        norm_i[i][3] = cam_3d(i, 3);
    }
    // Clipping dos valores fora do intervalo
    for (pybind11::ssize_t i = 0; i < num_points; ++i) {
        norm[i][0] = max(0.001f, min(0.999f, norm[i][0]));
        norm[i][1] = max(0.001f, min(0.999f, norm[i][1]));
    }

    // Escalonamento para o tamanho da imagem
    for (pybind11::ssize_t i = 0; i < num_points; ++i) {
        norm[i][0] *= image_width;
        norm[i][1] *= image_height;
    }

    // Definindo posições e índices
    vector<vector<int>> pos(num_points, vector<int>(2, 0));
    for (pybind11::ssize_t i = 0; i < num_points; ++i) {
        pos[i][0] = static_cast<int>(std::floor(norm[i][0]));
        pos[i][1] = static_cast<int>(std::floor(norm[i][1]));
    }
    vector<int> idx(num_points);
    for (pybind11::ssize_t i = 0; i < num_points; ++i) {
        idx[i] = static_cast<int>(i);
    }

    // Implementando o índice de cada posição
    vector<vector<int>> pos_idx(num_points, vector<int>(3, 0));
    for (pybind11::ssize_t i = 0; i < num_points; ++i) {
        pos_idx[i][0] = pos[i][0];
        pos_idx[i][1] = pos[i][1];
        pos_idx[i][2] = idx[i];
    }

    // Agrupando detecções na mesma célula da grade
    std::vector<std::pair<GridKey, vector<int>>> dic_pillar_ordered;
    // Construir o dicionário ordenado
    for (pybind11::ssize_t i = 0; i < num_points; ++i) {
        GridKey key = make_tuple(pos_idx[i][0], pos_idx[i][1]);

        // Verificar se a chave já existe
        auto it = std::find_if(dic_pillar_ordered.begin(), dic_pillar_ordered.end(),
                            [&key](const std::pair<GridKey, vector<int>>& pair) {
                                return pair.first == key;
                            });

        if (it != dic_pillar_ordered.end()) {
            // Adicionar o índice à lista existente se o tamanho máximo não foi atingido
            if (it->second.size() < static_cast<size_t>(max_group)) {
                it->second.push_back(pos_idx[i][2]);
            }
        } else {
            // Adicionar uma nova chave ao dicionário
            dic_pillar_ordered.emplace_back(key, vector<int>{pos_idx[i][2]});
        }
    }

    // Preparando arrays de saída
    int pillar_dim_0 = input_pillar_l_shape[0].cast<int>();
    int pillar_dim_1 = input_pillar_l_shape[1].cast<int>();
    int pillar_dim_2 = input_pillar_l_shape[2].cast<int>();

    py::array_t<float> vox_pillar({pillar_dim_0, pillar_dim_1, pillar_dim_2});
    std::fill(vox_pillar.mutable_data(), vox_pillar.mutable_data() + vox_pillar.size(), 0.0f);

    py::array_t<float> vox_pillar_indices({input_pillar_l_indices_shape[0].cast<int>(), input_pillar_l_indices_shape[1].cast<int>()});
    std::fill(vox_pillar_indices.mutable_data(), vox_pillar_indices.mutable_data() + vox_pillar_indices.size(), 0.0f);
    
    auto vox_pillar_mut = vox_pillar.mutable_unchecked<3>();
    auto vox_pillar_indices_mut = vox_pillar_indices.mutable_unchecked<2>();

    int j = 0;
    for (const auto& pair : dic_pillar_ordered) {
        if (j >= pillar_dim_0) {
            break;
        }

        GridKey key = pair.first;
        const vector<int>& v = pair.second;

        int k = 0;
        vector<float> vox_pillar_mean(3, 0.0f);

        for (int id : v) {
            for (int l = 0; l < 4; ++l) {
                vox_pillar_mut(j, k, l) = norm_i[id][l];
            }
            vox_pillar_mut(j, k, 7) = abs((norm[id][0] - std::get<0>(key)) - 0.5f);
            vox_pillar_mut(j, k, 8) = abs((norm[id][1] - std::get<1>(key)) - 0.5f);

            vox_pillar_mean[0] += norm_i[id][0];
            vox_pillar_mean[1] += norm_i[id][1];
            vox_pillar_mean[2] += norm_i[id][2];

            k += 1;
            if (k == pillar_dim_1) {
                break;
            }
        }

        if (k > 0) {
            vox_pillar_mean[0] /= k;
            vox_pillar_mean[1] /= k;
            vox_pillar_mean[2] /= k;

            for (int n = 0; n < k; ++n) {
                vox_pillar_mut(j, n, 4) = abs(vox_pillar_mut(j, n, 0) - vox_pillar_mean[0]);
                vox_pillar_mut(j, n, 5) = abs(vox_pillar_mut(j, n, 1) - vox_pillar_mean[1]);
                vox_pillar_mut(j, n, 6) = abs(vox_pillar_mut(j, n, 2) - vox_pillar_mean[2]);
            }
        }

        vox_pillar_indices_mut(j, 1) = static_cast<float>(std::get<0>(key));
        vox_pillar_indices_mut(j, 2) = static_cast<float>(std::get<1>(key));

        j += 1;
        if (j == max_pillars) {
            break;
        }
    }
    // for (auto it = dic_pillar.begin(); it != dic_pillar.end() && j < pillar_dim_0; ++it, ++j) {
    //     GridKey key = it->first;
    //     vector<int>& v = it->second;

    //     int k = 0;
    //     vector<float> vox_pillar_mean(3, 0.0f);

    //     for (int id : v) {
    //         for (int l = 0; l < 4; ++l) {
    //             vox_pillar_mut(j, k, l) = norm_i[id][l];
    //         }
    //         vox_pillar_mut(j, k, 7) = abs((norm[id][0] - std::get<0>(key)) - 0.5f);
    //         vox_pillar_mut(j, k, 8) = abs((norm[id][1] - std::get<1>(key)) - 0.5f);

    //         vox_pillar_mean[0] += norm_i[id][0];
    //         vox_pillar_mean[1] += norm_i[id][1];
    //         vox_pillar_mean[2] += norm_i[id][2];

    //         k += 1;
    //         if (k == pillar_dim_1) {
    //             break;
    //         }
    //     }

    //     if (k > 0) {
    //         vox_pillar_mean[0] /= k;
    //         vox_pillar_mean[1] /= k;
    //         vox_pillar_mean[2] /= k;

    //         for (int n = 0; n < k; ++n) {
    //             vox_pillar_mut(j, n, 4) = abs(vox_pillar_mut(j, n, 0) - vox_pillar_mean[0]);
    //             vox_pillar_mut(j, n, 5) = abs(vox_pillar_mut(j, n, 1) - vox_pillar_mean[1]);
    //             vox_pillar_mut(j, n, 6) = abs(vox_pillar_mut(j, n, 2) - vox_pillar_mean[2]);
    //         }
    //     }

    //     vox_pillar_indices_mut(j, 1) = static_cast<float>(std::get<0>(key));
    //     vox_pillar_indices_mut(j, 2) = static_cast<float>(std::get<1>(key));

    //     j += 1;
    //     if (j == max_pillars) {
    //         break;
    //     }
    // }

    return {vox_pillar, vox_pillar_indices};
}

PYBIND11_MODULE(vox_pillar_l_c_plus, m) {
    m.doc() = "Módulo de pilarização implementado em C++ corrigido para alinhamento com Python";
    m.def("pillaring_l", &pillaring_l, "Função de pilarização",
          py::arg("cam_3d_input"),
          py::arg("image_size"),
          py::arg("input_pillar_l_shape"),
          py::arg("input_pillar_l_indices_shape"),
          py::arg("max_group"),
          py::arg("max_pillars"),
          py::arg("x_min"), py::arg("x_diff"),
          py::arg("y_min"), py::arg("y_diff"),
          py::arg("z_min"), py::arg("z_diff"));
}

import numpy as np
import tensorflow as tf
import config_model as cfg
class ImageLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, validation_data, freq=5, X_div=None, Z_div=None):
        super().__init__()
        self.log_dir = log_dir
        self.validation_data = validation_data
        self.freq = freq
        self.X_div = X_div
        self.Z_div = Z_div
        self.writer = tf.summary.create_file_writer(log_dir + '/images')
        self.ground_truth_logged = False  # To control the logging of ground truth

    def on_train_begin(self, logs=None):
        """
        Carregamos todas as amostras de 'validation_data' na memória.
        Assim, podemos fazer a predição e o log de todas de uma só vez.
        """
        self.all_val_pL   = []
        self.all_val_pos  = []
        self.all_val_imgs = []
        self.all_val_labels = []

        # Percorre todo o validation_data (que deve ser iterável)
        for X_batch, y_batch in self.validation_data:
            # Supondo que X_batch = [X_pL, X_pos, X_imgs], e a imagem propriamente dita seja X_batch[2]
            X_pL, X_posL, X_imgs = X_batch  # separa cada componente
            self.all_val_pL.append(X_pL)
            self.all_val_pos.append(X_posL)
            self.all_val_imgs.append(X_imgs)
            self.all_val_labels.append(y_batch)

        # Concatena tudo num único array
        self.all_val_pL    = np.concatenate(self.all_val_pL,   axis=0)
        self.all_val_pos   = np.concatenate(self.all_val_pos,  axis=0)
        self.all_val_imgs  = np.concatenate(self.all_val_imgs, axis=0)
        self.all_val_labels = np.concatenate(self.all_val_labels, axis=0)

        print(f"Total de imagens de validação mini carregadas: {len(self.all_val_imgs)}")

    def on_epoch_end(self, epoch, logs=None):
        """
        A cada epoch, se freq for atingido, rodamos o predict em todas as amostras.
        """

        if epoch % self.freq == 0:  # Log every 'freq' epochs
            val_images, val_labels = next(iter(self.validation_data))
            # dt = self.model.predict(self.all_val_images, batch_size=cfg.BATCH_SIZE)  # Predict the whole batch
            dt = self.model.predict([self.all_val_pL, self.all_val_pos, self.all_val_imgs],
                                    batch_size=cfg.BATCH_SIZE
                                    )
            with self.writer.as_default():
                # Loga quantas tivermos (todas)
                num_to_log = len(self.all_val_imgs)
                for i in range(num_to_log):
                    # Reshape and prepare the prediction image FOR 2 ANCHORS OR ONE CLASS OBJ
                    # occupancy = np.reshape(dt[i, ..., 0], (self.X_div, self.Z_div, 2))

                    # 1) Pega as âncoras 0 e 1, soma para dar shape (X_div, Z_div)
                    occupancy_1 = dt[i, :, :, 0:2, 0]  # (X_div, Z_div, 2, 1)
                    # print(occupancy_1.shape)
                    occupancy_1_sum = np.sum(occupancy_1, axis=-1, keepdims=True)  # (X_div, Z_div, 1, 1)
                    # print(occupancy_1_sum.shape)
                    # exit()
                    # occupancy_1_sum = np.reshape(occupancy_1_sum[i, ...], (self.X_div, self.Z_div, 1))

                    # 2) Pega as âncoras 2 e 3, soma
                    occupancy_2 = dt[i, :, :, 2:4, 0]  # (X_div, Z_div, 2, 1)
                    occupancy_2_sum = np.sum(occupancy_2, axis=-1, keepdims=True)  # (X_div, Z_div, 1, 1)
                    # occupancy_2_sum = np.reshape(occupancy_2_sum[i, ...], (self.X_div, self.Z_div, 1))

                    occupancy = np.concatenate((occupancy_1_sum, occupancy_2_sum), axis=-1)
                    img = np.concatenate((occupancy, np.zeros((self.X_div, self.Z_div, 1))), axis=-1)

                    tf.summary.image(f"Epoch:{epoch}_Prediction_{i}",
                                     np.expand_dims(img, axis=0),
                                     step=epoch)

                # Log ground truth once
                if not self.ground_truth_logged:
                    for j in range(len(self.all_val_imgs)):
                        conf_matrix = np.reshape(self.all_val_labels[j, ..., 0], (self.X_div, self.Z_div, 4))
                        conf_matrix1 = conf_matrix[...,:2]
                        conf_matrix1 = np.sum(conf_matrix1, axis=-1, keepdims=True)
                        conf_matrix2 = conf_matrix[...,2:]
                        conf_matrix2 = np.sum(conf_matrix2, axis=-1, keepdims=True)

                        conf_matrix = np.concatenate((conf_matrix1, conf_matrix2), axis=-1)

                        conf_matrix_img = np.concatenate((conf_matrix, np.zeros((self.X_div, self.Z_div, 1))), axis=-1)
                        tf.summary.image(f"Ground_Truth_{j}",
                                         np.expand_dims(conf_matrix_img, axis=0),
                                         step=0)
                    self.ground_truth_logged = True  # Prevent further logging of ground truth

            self.writer.flush()  # Ensure logs are written

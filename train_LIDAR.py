import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import datetime
from models.myModel_pillar_1_old import My_Model
from dataset_LIDAR import SequenceData
# from Pillar_loss_M import PointPillarNetworkLoss
from Loss_LIDAR import Loss, focal_loss, loc_loss, size_loss, angle_loss
import numpy as np
import config_model as cfg
from accuracy import correct_grid, incorrect_grid

EPOCHS = 360
BATCH_SIZE = cfg.BATCH_SIZE
ITERS_TO_DECAY = cfg.ITERS_TO_DECAY
LEARNING_RATE = cfg.LEARNING_RATE
DECAY_RATE = cfg.DECAY_RATE

input_pillar_l_shape = cfg.input_pillar_l_shape
input_pillar_pos_l_shape = cfg.input_pillar_l_indices_shape

# input_pillar_r_shape = cfg.input_pillar_r_shape
# input_pillar_pos_r_shape = cfg.input_pillar_r_indices_shape

# input_img_shape = cfg.img_shape
img_shape = cfg.img_shape

tf.get_logger().setLevel("ERROR")




DATASET_PATH = cfg.DATASET_PATH
# LABEL_PATH = cfg.LABEL_PATH


def train():
	batch_size = BATCH_SIZE

	input_pillar_l = Input(input_pillar_l_shape, batch_size=batch_size)
	input_pillar_pos_l = Input(input_pillar_pos_l_shape, batch_size=batch_size)

	# input_pillar_r = Input(input_pillar_r_shape, batch_size=batch_size)
	# input_pillar_pos_r = Input(input_pillar_pos_r_shape, batch_size=batch_size)

	# input_img = Input((img_shape[0],img_shape[1],3),batch_size = batch_size)

	output = My_Model(input_pillar_l, input_pillar_pos_l)
	model = Model(inputs=[input_pillar_l, input_pillar_pos_l], outputs=output)


	model.load_weights(os.path.join('checkpoints/val_loss_LIDAR/Temp_loss/', "model_050_Model_pillar.hdf5"))
	print('Model Loaded!\n')
	#########################################
	#										#
	#               COMPILE                 #
	#									    #
	#########################################

	# loss = PointPillarNetworkLoss()

	optimizer = Adam(learning_rate = LEARNING_RATE,weight_decay=0.0001)
	# optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE,
	# 									    weight_decay=0.01,
	# 									    beta_1=0.95,
	# 									    beta_2=0.99)

	model.compile(optimizer = optimizer, loss=Loss, metrics =[ correct_grid, incorrect_grid ,
															 focal_loss, loc_loss, size_loss,
															 angle_loss])
	# model.compile(optimizer = optimizer, loss=loss.losses())

	#########################################
	#										#
	#             CHECKPOINTS               #
	#                LOSS                   #
	#									    #
	#########################################

	save_dir_l = 'checkpoints/val_loss_LIDAR/'
	weights_path_l = os.path.join(save_dir_l,(datetime.datetime.now().strftime("%Y%m%d-") +'Model_pillar_1.hdf5'))
	checkpoint_loss = ModelCheckpoint(weights_path_l, monitor = 'val_loss',mode='min', save_best_only = True)

	

	#########################################
	#										#
	#              CALLBACKS                # 
	#									    #
	#########################################

	# early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 15, verbose = 1, mode = 'auto')

	log_dir = 'logs_LIDAR/'+ datetime.datetime.now().strftime("%Y%m%d-") + 'Model_pillar_1_C'
	tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_grads = True)


	#---------------------------------------------------------------------------------------	
	#########################################
	#										#
	#               DATASET                 # 
	#									    #
	#########################################

	dataset_path = DATASET_PATH
	''' Insert SequenceData '''
	train_gen = SequenceData('train', dataset_path, img_shape, batch_size,data_aug = False)
	# print(len(train_gen))
	# exit()

	valid_gen = SequenceData('val', dataset_path, img_shape, batch_size,data_aug = False)

	#########################################
	#										#
	#             CHECKPOINTS               #
	#                LOSS epoc              #
	#									    #
	#########################################checkpoint = keras.callbacks.ModelCheckpoint('model{epoch:08d}.h5', period=5) 
	checkpoint_loss_e = ModelCheckpoint('checkpoints/val_loss_LIDAR/Temp_loss/model_{epoch:03d}_Model_pillar.hdf5',
										save_weights_only = True, 
										save_freq = int(10*len(train_gen)))


	#---------------------------------------------------------------------------------------	
	#########################################
	#										#
	#              LR SCHEDULE              # 
	#									    #
	########################################

	def scheduler(epoch, lr):
		if epoch % 15 == 0 and epoch != 0:
			lr = lr*0.8
			return lr
		else:
			return lr

	lr_schd = tf.keras.callbacks.LearningRateScheduler(scheduler)
	#---------------------------------------------------------------------------------------	
	#########################################
	#										#
	#        MODEL FIT GENERATOR            # 
	#									    #
	########################################
	# model.summary()
	# exit()

	# label_files = os.listdir(LABEL_PATH)
	# epoch_to_decay = int(
 #        np.round(ITERS_TO_DECAY / BATCH_SIZE * int(np.ceil(float(len(label_files)) / BATCH_SIZE))))
	callbacks=[
				tbCallBack,
				checkpoint_loss,
				lr_schd,
				checkpoint_loss_e
					  ]
	# try:
	model.fit(
		train_gen,
		epochs = EPOCHS,
		validation_data=valid_gen,
		steps_per_epoch=len(train_gen),
		callbacks=callbacks,
		initial_epoch=50,
		use_multiprocessing = True,
		workers = 2
		)
	# except KeyboardInterrupt:
	# 	model.save('checkpoints/interrupt/Interrupt_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.hdf5')
	# 	print('Interrupt. Output saved')

	# try:
	# 	model.fit(
	# 		train_gen,




	# 		epochs = EPOCHS,
	# 		validation_data=valid_gen,
	# 		steps_per_epoch=len(train_gen),
	# 		callbacks=callbacks,
	# 		use_multiprocessing = True,
	# 		workers = 4
	# 		)
	# except KeyboardInterrupt:
	model.save('FINAL-'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.hdf5')
	# 	print('Interrupt. Output saved')

if __name__=='__main__':
	train()
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from image_callback import ImageLoggingCallback
import tensorflow as tf
import datetime
from models.myModel_Fusion_4 import My_Model
from dataset import SequenceData
# from Pillar_loss_M import PointPillarNetworkLoss
from Loss import Loss, focal_loss, loc_loss, size_loss, angle_loss
import numpy as np
import config_model as cfg
from accuracy import correct_grid, incorrect_grid

EPOCHS = 360
BATCH_SIZE = cfg.BATCH_SIZE
ITERS_TO_DECAY = cfg.ITERS_TO_DECAY
LEARNING_RATE = cfg.LEARNING_RATE
DECAY_RATE = cfg.DECAY_RATE

input_pillar_l_shape = cfg.input_pillar_shape
input_pillar_pos_l_shape = cfg.input_pillar_indices_shape


input_img_shape = cfg.image_shape
img_shape = cfg.image_shape

tf.get_logger().setLevel("ERROR")


from tensorflow.keras.mixed_precision import set_global_policy

# Configurar mixed precision
set_global_policy('mixed_float16')


DATASET_PATH = cfg.DATASET_PATH
# LABEL_PATH = cfg.LABEL_PATH


def train():
	batch_size = BATCH_SIZE

	input_pillar_l = Input(input_pillar_l_shape, batch_size=batch_size)
	input_pillar_pos_l = Input(input_pillar_pos_l_shape, batch_size=batch_size)

	input_img = Input((img_shape[0],img_shape[1],3),batch_size = batch_size)

	output = My_Model(input_pillar_l, input_pillar_pos_l,  input_img)
	model = Model(inputs=[input_pillar_l, input_pillar_pos_l, input_img], outputs=output)


	# model.load_weights(os.path.join('checkpoints/val_loss/Temp_loss/', "model_010_Model_minimum.hdf5"))
	# print('Model Loaded!\n')
	#########################################
	#										#
	#               COMPILE                 #
	#									    #
	#########################################

	# loss = PointPillarNetworkLoss()

	optimizer = Adam(learning_rate = LEARNING_RATE, clipnorm=1.0)
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

	save_dir_l = 'checkpoints/val_loss/'
	weights_path_l = os.path.join(save_dir_l,(datetime.datetime.now().strftime("%Y%m%d-") +'Model_Fusion_4_LC'))
	checkpoint_loss = ModelCheckpoint(weights_path_l, monitor = 'val_loss',mode='min', save_best_only = True)

	

	#########################################
	#										#
	#              CALLBACKS                # 
	#									    #
	#########################################

	# early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 15, verbose = 1, mode = 'auto')

	log_dir = 'logs/'+ datetime.datetime.now().strftime("%Y%m%d-") + 'Model_Fusion_4_LC'
	tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_grads = True)


	#---------------------------------------------------------------------------------------	
	#########################################
	#										#
	#               DATASET                 # 
	#									    #
	#########################################

	dataset_path = DATASET_PATH
	''' Insert SequenceData '''
	train_gen = SequenceData('train', dataset_path, img_shape, batch_size,data_aug = True)
	# print(len(train_gen))
	# exit()

	valid_gen = SequenceData('val', dataset_path, img_shape, batch_size,data_aug = False)

	valid_min_gen = SequenceData('val_mini', dataset_path, img_shape, batch_size,data_aug = False)

	#########################################
	#										#
	#             CHECKPOINTS               #
	#                LOSS epoc              #
	#									    #
	#########################################checkpoint = keras.callbacks.ModelCheckpoint('model{epoch:08d}.h5', period=5) 
	checkpoint_loss_e = ModelCheckpoint('checkpoints/val_loss/Temp_loss/model_{epoch:03d}_Model_Fusion_4_LC',
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
	image_logger = ImageLoggingCallback(log_dir=log_dir, validation_data=valid_min_gen, freq=3, X_div=cfg.X_div, Z_div=cfg.Z_div)

	# label_files = os.listdir(LABEL_PATH)
	# epoch_to_decay = int(
 #        np.round(ITERS_TO_DECAY / BATCH_SIZE * int(np.ceil(float(len(label_files)) / BATCH_SIZE))))
	callbacks=[
				tbCallBack,
				checkpoint_loss,
				lr_schd,
				checkpoint_loss_e,
				image_logger
					  ]
	# try:
	model.fit(
		train_gen,
		epochs = EPOCHS,
		validation_data=valid_gen,
		steps_per_epoch=len(train_gen),
		callbacks=callbacks,
		# initial_epoch=10,
		# use_multiprocessing = True,
		# workers = 2
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
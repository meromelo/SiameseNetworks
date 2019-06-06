# Implementation of SqueezeNet
#
# SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
# https://arxiv.org/abs/1602.07360
#
# Identity Mappings in Deep Residual Networks
# https://arxiv.org/abs/1603.05027)# Wide Residual Network

#
# Standard library imports.
#
from __future__ import absolute_import, print_function, unicode_literals

import os
import os.path as path
import subprocess
import sys
import argparse
import glob
import argparse
import tempfile
import shutil
import lzma
from functools import partial
from io import StringIO

#
#	plaidml
#
import plaidml.keras
plaidml.keras.install_backend()

#
# keras
#
import keras
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
from keras						import backend as K
from keras						import models
from keras.callbacks			import LearningRateScheduler
from keras.layers				import Add, Conv2D, Dense, MaxPooling2D, Activation, Flatten, Dropout, Lambda, ELU, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, Concatenate, MaxPooling2D
from keras.layers.convolutional	import Convolution2D
from keras.layers.pooling		import MaxPooling2D, AveragePooling2D
from keras.models				import Model, Sequential, save_model, model_from_json
from keras.optimizers			import Adam, RMSprop, SGD
from keras.preprocessing.image	import ImageDataGenerator
from keras.regularizers			import l2
from keras.utils				import plot_model
from keras.activations			import relu, softmax
from keras.callbacks			import EarlyStopping

#
# Image, picke, numpy, matplotlib
#
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt

#
# tensorflow
#
import tensorflow as tf

#
# sklearn
#
import sklearn
from sklearn.manifold			import TSNE
from sklearn.decomposition		import PCA

#
# funcy
#
from funcy			import concat, identity, juxt, partial, rcompose, repeat, take

#
# hyperdash
#
from hyperdash		import Experiment as hyperdash_experiement
from lib_observe	import hyperdash_callback

#
# Local application/library specific imports.
#
from lib_CIFAR_10	import CIFAR_10
from operator		import getitem, add, mul

#
# network
#
from networks		import SqueezeNet

#
# type annotation (PEP 484) >= Python 3.5
#	https://docs.python.org/3/library/typing.html
#	https://www.hacky.xyz/entry/20180819/python-typehints
#	https://note.nkmk.me/python-function-annotations-typing/
#
#	1. Dict[str,int], Tuple[str, int], List[str]
#		def dosomething() -> Union[str, int]:
#			# returns int or str
#	2. Optional型 : Union[A, None] と同じ意味（Optional[X] = Union[X, None]）
#		from typing import Optional
#		def convert_to_int(value: string) -> Optonal[int]:
#			# returns int or None
#	3. own class : 
#		class UserModel():
#			@classmethod
#			def get_user_from_session() -> 'UserModel':
#				# returns UserModel object
#	4. List[Union[X,Y]]
#		from typing import Union, List
#		def func_u(x: List[Union[int, float]]) -> float:
#			return sum(x) ** 0.5
#		print(func_u([0.5, 9.5, 90]))
#		# 10.0
#	5. Callable
#		Callable[[int],str] = （int） - > str
#		def call(func: Callable[[str, int], str]) -> None:
#
from typing import Union, Dict, Tuple, List, Optional, Callable
tensor4D = Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]

# FS_ENCODING = sys.getfilesystemencoding()

#
#	model path
#
model_path	= "./results/"
if not path.exists(model_path):
	os.mkdir(model_path)

#####################
#
# CLASS
# Input preprocessing.
# Here we create some functions that will create the input couple for our model, both correct and wrong couples.
# I created functions to have both depth-only input and RGBD inputs.
#
#####################
def train():
	#
	# make matrix (input data)
	#
	# def __make_mat(file_path:str, folder_path:str="")->tuple(str,np.ndarray,np.ndarray,np.ndarray):
	# def __make_mat(file_path:str, folder_path:str="")->tuple[str,np.ndarray,np.ndarray,np.ndarray]:
	def __make_mat(file_path:str, folder_path:str="")->Tuple[str,np.ndarray,np.ndarray,np.ndarray]:
		#
		# params
		#
		mat = np.zeros((480, 640), dtype='float32')
		i = 0
		j = 0

		#
		# folder & depth_file
		#
		if len(folder_path) == 0:
			folder_path = np.random.choice(glob.glob(file_path + "*"))

		depth_file = np.random.choice(glob.glob(folder_path + "/*.dat"))
		# print( f"__make_mat: folder_path={folder_path}, file_path={file_path}" )

		with open(depth_file) as file:
			for line in file:
				vals = line.split('\t')
				for val in vals:
					if val == "\n": continue
					if int(val) > 1200 or int(val) == -1: val = 1200
					mat[i][j] = float(int(val))
					j += 1
					j = j % 640
				i += 1
			mat = np.asarray(mat)
		mat_small  = mat[140:340, 220:420]
		mat_smallr = (mat_small - np.mean(mat_small)) / np.max(mat_small)

		#
		# rgbd
		#
		img = Image.open(depth_file[:-5] + "c.bmp")
		img.thumbnail((640, 480))
		img = np.asarray(img)
		img = img[160:360, 240:440]

		full = np.zeros((200, 200, 4))
		full[:, :, :3] = img[:, :, :3]
		full[:, :,  3] = mat_smallr

		return (folder_path, mat_small, mat_smallr, full)



	#####################
	#
	# create couple
	#
	def create_couple(file_path:str)->np.ndarray:
		(folder_path, _, mat1, _) = __make_mat(file_path)
		(folder_path, _, mat2, _) = __make_mat(file_path, folder_path)
		return np.array([mat1, mat2])

	#####################
	#
	# create couple rgbd
	#
	def create_couple_rgbd(file_path:str)->np.ndarray:
		(folder_path, _, _, full1) = __make_mat(file_path)
		(folder_path, _, _, full2) = __make_mat(file_path, folder_path)
		return np.array([full1, full2])

	#####################
	#
	# create wrong
	#
	def create_wrong(file_path:str)->np.ndarray:
		(_, _, mat1, _) = __make_mat(file_path)
		(_, _, mat2, _) = __make_mat(file_path)
		return np.array([mat1, mat2])

	#####################
	#
	# create wrong rgbd
	#
	def create_wrong_rgbd(file_path:str)->np.ndarray:
		(_, _, _, full1) = __make_mat(file_path)
		(_, _, _, full2) = __make_mat(file_path)
		return np.array([full1, full2])

	#####################
	#
	# euclidean distance
	#
	def euclidean_distance(inputs:Tuple[float,float])->float:
		assert len(inputs) == 2, 'Euclidean distance needs 2 inputs, %d given' % len(inputs)
		u, v = inputs
		return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))

	#####################
	#
	# contrastive loss
	#
	def contrastive_loss(y_true:float, y_pred:float)->float:
		margin = 1.
		return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

	#####################
	#
	# fire
	#
#	tensor4D = Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]
#	def fire(x:Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray], squeeze:int=16, expand:int=64)->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
	def fire(x:tensor4D, squeeze:int=16, expand:int=64)->tensor4D:
		x		= Convolution2D(squeeze, (1, 1), padding='valid')(x)
		x		= Activation('relu')(x)

		left	= Convolution2D(expand, (1, 1), padding='valid')(x)
		left	= Activation('relu')(left)

		right	= Convolution2D(expand, (3, 3), padding='same')(x)
		right	= Activation('relu')(right)

		x		= keras.backend.concatenate([left,right], axis=3)

		return x

	#####################
	#
	# input_generator
	#
	def __input_generator(batch_size, file_path="RGB-D_Face_database/faceid_train/"):
		while 1:
			x = []
			y = []
			# print("generator0:")
			# print(x)
			# print(y)
			switch = True
			for _ in range(batch_size):
				if switch:
					x.append(create_couple_rgbd(file_path).reshape((2, 200, 200, 4)))
					y.append(np.array([0.]))
				else:
					x.append(create_wrong_rgbd (file_path).reshape((2, 200, 200, 4)))
					y.append(np.array([1.]))
				switch = not switch
			x = np.asarray(x)
			y = np.asarray(y)
			# XX1 = x[0, :]
			# XX2 = x[1, :]
			# print("input_generator:")
			# print(x)
			# print(y)
			yield [x[:, 0], x[:, 1]], y

	#####################
	#
	# valid_generator (generator for validation)
	#
	def __valid_generator(batch_size, file_path="RGB-D_Face_database/faceid_val/"):
		while 1:
			x = []
			y = []
			# print("val_generator0:")
			# print(x)
			# print(y)
			switch = True
			for _ in range(batch_size):
				if switch:
					x.append(create_couple_rgbd(file_path).reshape((2, 200, 200, 4)))
					y.append(np.array([0.]))
				else:
					x.append(create_wrong_rgbd (file_path).reshape((2, 200, 200, 4)))
					y.append(np.array([1.]))
				switch = not switch
			x = np.asarray(x)
			y = np.asarray(y)
			# XX1 = x[0, :]
			# XX2 = x[1, :]
			# print("valid_generator:")
			# print(x)
			# print(y)
			yield [x[:, 0], x[:, 1]], y

	#####################
	#
	# run
	#
	def run():
		#
		#	hyperdash
		#
		hye = hyperdash_experiement("faceid")
		hye_callback = hyperdash_callback(exp=hye)

		# print(self.create_couple("RGB-D_Face_database/faceid_train/"))
		# print(self.create_couple_rgbd("RGB-D_Face_database/faceid_val/"))
		create_couple_rgbd("RGB-D_Face_database/faceid_val/")
		# print(self.create_wrong("RGB-D_Face_database/faceid_train/"))
		# print(self.create_wrong_rgbd("RGB-D_Face_database/faceid_val/")[0].shape)

		# quit()

		#
		# create network
		#
		img_input = Input(shape=(200, 200, 4))

		x = Convolution2D(64, (5, 5), strides=(2, 2), padding='valid')(img_input)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

		x = fire(x, squeeze=16, expand=16)
		x = fire(x, squeeze=16, expand=16)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

		x = fire(x, squeeze=32, expand=32)
		x = fire(x, squeeze=32, expand=32)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

		x = fire(x, squeeze=48, expand=48)
		x = fire(x, squeeze=48, expand=48)
		x = fire(x, squeeze=64, expand=64)
		x = fire(x, squeeze=64, expand=64)
		x = Dropout(0.2)(x)

		x = Convolution2D(512, (1, 1), padding='same')(x)
		out = Activation('relu')(x)

		modelsqueeze = Model(img_input, out)
		print("\nmodel squeeze summary")
		modelsqueeze.summary()
		plot_model(modelsqueeze, show_shapes=True, to_file='model_squeeze.png')

		im_in = Input(shape=(200, 200, 4))
		x1 = modelsqueeze(im_in)
		x1 = Flatten()(x1)
		x1 = Dense(512, activation="relu")(x1)
		x1 = Dropout(0.2)(x1)
		feat_x = Dense(128, activation="linear")(x1)
		feat_x = Lambda(lambda x: K.l2_normalize(x, axis=1))(feat_x)

		model_top = Model(inputs=[im_in], outputs=feat_x)
		print("\nmodel top summary")
		model_top.summary()
		plot_model(model_top, show_shapes=True, to_file='model_top.png')

		im_in1 = Input(shape=(200, 200, 4))
		im_in2 = Input(shape=(200, 200, 4))
		feat_x1 = model_top(im_in1)
		feat_x2 = model_top(im_in2)
		lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])

		model_final = Model(inputs=[im_in1, im_in2], outputs=lambda_merge)
		print("\nmodel final summary")
		model_final.summary()
		plot_model(model_final, show_shapes=True, to_file='model_final.png')

		adam = Adam(lr=0.001)
		sgd = SGD(lr=0.001, momentum=0.9)

		model_final.compile(optimizer=adam, loss=contrastive_loss)

		#
		# plot model
		#
		# print("write model summary png...")
		# plot_model(model_final, show_shapes=True, to_file='model.png')
		# print("write model summary png...done")

		#
		# input_generator
		#
		input_generator = __input_generator(16)
		valid_generator = __valid_generator(4)

		#
		# checkpoint
		# 各エポック終了後にモデルを保存
		# 	file_name = str(datetime.datetime.now()).split(' ')[0] + '_{epoch:02d}.hdf5'
		# 	filepath = os.path.join(save_dir, file_name)
		#
		"""
		keras.callbacks.ModelCheckpoint(
			filepath,
			monitor='val_loss',
			verbose=0,
			save_best_only=False,
			save_weights_only=False,
			mode='auto',
			period=1)
		"""
		drive_dir = 'RGB-D_Face_database/snapshot/'
		base_file_name = 'model'
		checkpointer = keras.callbacks.ModelCheckpoint(
			# filepath=drive_dir+base_file_name+'.{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5',
			filepath=drive_dir + base_file_name + '.epoch{epoch:03d}-loss{loss:.4f}-val_loss{val_loss:.4f}.hdf5',
			# filepath=drive_dir+base_file_name+'.{epoch:02d}-{val_loss:.2f}.hdf5',
			verbose=1,
			save_best_only=True,
			# monitor='val_acc',
			monitor='val_loss',
			mode='auto')

		#
		# ProgressbarLogger
		#
		pbarl = keras.callbacks.ProgbarLogger(count_mode='samples');

		#
		# CSV Logger
		# 各エポックの結果をcsvファイルに保存する (Google Driveでは学習終了まで反映されない。localに保存)
		#
		"""
		keras.callbacks.CSVLogger(
			filename,
			separator=',',
			append=False)
		"""
		csv_logger = keras.callbacks.CSVLogger('./xxx.log')

		#
		# reduce LR on plateau
		# 評価値の改善が止まった時に学習率を減らす
		#
		"""
		keras.callbacks.ReduceLROnPlateau(
			monitor='val_loss',
			factor=0.1,
			patience=10,
			verbose=0,
			mode='auto',
			epsilon=0.0001,
			cooldown=0,
			min_lr=0)
		"""
		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

		#
		# early stopping
		#
		"""
		keras.callbacks.EarlyStopping(
			monitor='val_loss',
			min_delta=0,
			patience=0,
			verbose=0,
			mode='auto')
		"""
		early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=100)

		#
		# tensor board
		#
		"""
		keras.callbacks.TensorBoard(
			log_dir='./logs',
			histogram_freq=0,
			batch_size=32,
			write_graph=True,
			write_grads=False,
			write_images=False,
			embeddings_freq=0,
			embeddings_layer_names=None,
			embeddings_metadata=None)
		$tensorboard --logdir=/full_path_to_your_logs
		"""
		# tensorboard = keras.callbacks.TensorBoard(log_dir="RGB-D_Face_database/log", histogram_freq=1)
		# old_session = KTF.get_session()
		# new_session = tf.Session('')
		# KTF.set_session(new_session)

		#
		# input_generator
		#
		"""
		outputs = model_final.fit_generator(
			generator, 
			steps_per_epoch=None, 
			epochs=1, 
			verbose=1, 
			callbacks=None, 
			validation_data=None, 
			validation_steps=None, 
			class_weight=None, 
			max_queue_size=10, 
			workers=1, 
			use_multiprocessing=False, 
			shuffle=True, 
			initial_epoch=0)
		"""
		# steps_per_epoch=30,
		# epochs=50,
		# callbacks=[checkpointer, csv_logger, reduce_lr, early_stop, tensorboard, hd_callback],
		# validation_steps=20
		# fit_generator(
		#	self,
		#	generator,
		#	steps_per_epoch=None,
		#	epochs=1,
		#	verbose=1,
		#	callbacks=None,
		#	validation_data=None,
		#	validation_steps=None,
		#	class_weight=None,
		#	max_queue_size=10,
		#	workers=1,
		#	use_multiprocessing=False,
		#	shuffle=True,
		#	initial_epoch=0)
		outputs = model_final.fit_generator(
			input_generator,
			steps_per_epoch=10,		# 30
			epochs=1,				# 50
			verbose=1,
			# callbacks=[checkpointer],
			# callbacks=[checkpointer, hd_callback],
			callbacks=[checkpointer, csv_logger, early_stop, reduce_lr, hye_callback],
			# callbacks=[checkpointer, pbarl, csv_logger, early_stop, reduce_lr, hd_callback],
			# callbacks=[checkpointer, csv_logger, early_stop, reduce_lr, tensorboard, hd_callback],
			# pickle_safe=True,
			validation_data=valid_generator,
			validation_steps=20,
			# workers=8,
			use_multiprocessing=True)		# 20

		#
		# model save
		#
		print('saving model_final...')
		model_final.save("RGB-D_Face_database/snapshot/model_final.h5")
		print('saving model_final...done')

		#
		# model test
		#
		"""
		"""
		file_path_for_validation = "RGB-D_Face_database/faceid_val/"

		cop = create_couple(file_path_for_validation)
		score = model_final.evaluate([cop[0].reshape((1, 200, 200, 4)), cop[1].reshape((1, 200, 200, 4))], np.array([0.]))
		print('Test score(couple):', score[0])
		print('Test accuracy(couple):', score[1])

		cop = create_wrong_rgbd(file_path_for_validation)
		score = model_final.predict([cop[0].reshape((1, 200, 200, 4)), cop[1].reshape((1, 200, 200, 4))])
		print('Test score(wrong_rgbd):', score[0])
		print('Test accuracy(wrong_rgbd):', score[1])

		#
		# save model (architecture,json)
		#
		print('save the architecture of a model...')
		json_string = model_final.to_json()
		open(drive_dir + base_file_name + 'model.json', 'w').write(json_string)
		# open(os.path.join(drive_dir+base_file_name,'model.json'), 'w').write(json_string)
		print('save the architecture of a model...done')

		print('save weights...')
		yaml_string = model_final.to_yaml()
		open(drive_dir + base_file_name + 'model.yaml', 'w').write(yaml_string)
		# open(os.path.join(drive_dir+base_file_name,'model.yaml'), 'w').write(yaml_string)
		model_final.save_weights(drive_dir + base_file_name + 'model_weights.hdf5')
		# model_final.save_weights(os.path.join(drive_dir+base_file_name,'model_weights.hdf5'))
		print('save weights...done')

		# debug
		print('debug: load_model...')
		del model_final
		model_final = keras.models.load_model(
			"RGB-D_Face_database/snapshot/model_final.h5",
			# custom_objects={
			# 'euclidean_distance': euclidean_distance,
			# 'contrastive_loss': contrastive_loss,
			# 'l2_normalize': K.l2_normalize
			# },
			compile=False)
		print('debug: load_model...done')

		#
		# tensorboard
		#
		# KTF.set_session(old_session)
		# print('tensorboard done')

		#
		# hyperdash
		#
		print('hyperdash done')
		hye.end()

	#####################
	#
	# data_augmentation
	# this will do preprocessing and realtime data augmentation
	#
	def	data_augmentation(self):
		datagen = ImageDataGenerator(
			featurewise_center=False,				# set input mean to 0 over the dataset
			samplewise_center=False,				# set each sample mean to 0
			featurewise_std_normalization=False,	# divide inputs by std of the dataset
			samplewise_std_normalization=False,		# divide each input by its std
			zca_whitening=False,					# apply ZCA whitening
			rotation_range=0,						# randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=0.1,					# randomly shift images horizontally (fraction of total width)
			height_shift_range=0.1,					# randomly shift images vertically (fraction of total height)
			horizontal_flip=True,					# randomly flip images
			vertical_flip=False						# randomly flip images
		)
		return datagen

	#####################
	#
	# predict
	#
	def predict(self):
		from keras.models import load_model
		from keras.models import model_from_json
		from keras.layers import Activation

		# Required, as usual
		from keras.models import load_model

		# Recommended method; requires knowledge of the underlying architecture of the model
		# from keras_contrib.layers.advanced_activations import PELU

		# Not recommended; however this will correctly find the necessary contrib modules
		#	from keras_contrib import *

		model_file = 'model_final.h5'
		print('current keras version:' + keras.__version__)

		import h5py
		f = h5py.File(model_file, 'r')
		print('model   keras version:' + f.attrs.get('keras_version').decode('utf-8'))

		"""
		# 1.
		json_filename = "arch.json"
		weight_filename = "weights.h5"
		#	json_string = open(os.path.join(f_model, model_filename)).read()
		json_string = open(json_filename).read()
		model = keras.models.model_from_json(json_string)
		model.summary()

		model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5), metrics=['accuracy'])
		#	model.load_weights(os.path.join(f_model,weights_filename))
		model.keras.models.load_weights(weights_filename)
		"""

		#
		# custom objects
		#   lambda_merge = Lambda(self.euclidean_distance)([feat_x1, feat_x2])
		#   feat_x = Lambda(lambda x: K.l2_normalize(x, axis=1))(feat_x)
		#
		# from keras.utils.generic_utils import get_custom_objects
		#   get_custom_objects().update({"SSD_Loss": loss.computeloss})
		# custom_objects = {'<lambda>': lambda x: K.l2_normalize(x, axis=1)}

		# 2.
		#   feat_x = Lambda(lambda x: K.l2_normalize(x, axis=1))(feat_x)
		# lambda_merge = Lambda(self.euclidean_distance)([feat_x1, feat_x2])

		# from keras.utils import CustomObjectScope
		# with CustomObjectScope({'lambda': K.l2_normalize}):
		# 	model = keras.models.load_model(model_file, compile=False)

		"""
		"""
		model = keras.models.load_model(
			model_file,
			custom_objects={
			'l2_normalize': K.l2_normalize,
			'euclidean_distance': euclidean_distance,
			'contrastive_loss': contrastive_loss
			},
			compile=False)
		"""
		"""
		print('model read done.')

		"""
		im_in1 = Input(shape=(200, 200, 4))
		feat_x1 = model_top(im_in1)


		model_output = Model(inputs=im_in1, outputs=feat_x1)
		model_output.summary()

		adam = Adam(lr=0.001)
		sgd = SGD(lr=0.001, momentum=0.9)
		model_output.compile(optimizer=adam, loss=self.contrastive_loss)
		"""

		print('try couple_rgbd, and predict.')
		cop = self.create_couple_rgbd(self, "RGB-D_Face_database/faceid_val/")
		# model_output.predict(cop[0].reshape((1, 200, 200, 4)))
		model.predict(cop[0].reshape((1, 200, 200, 4)))

	#####################
	#
	# create_input_rgbd
	#
	def create_input_rgbd(self, file_path):
		(folder_path, _, mat, full) = __make_mat(file_path)
		return np.array([full])

	#####################
	#
	# t-SNE
	#
	def tsne(self):
		outputs = []
		n = 0
		for folder in glob.glob('RGB-D_Face_database/faceid_train/*'):
			i = 0
			for file in glob.glob(folder + '/*.dat'):
				i += 1
				outputs.append(model_output.predict(create_input_rgbd(file).reshape((1, 200, 200, 4))))
				print(i)
			n += 1
		print("Folder ", n, " of ", len(glob.glob('RGB-D_Face_database/faceid_train/*')))
		print(len(outputs))

		outputs = np.asarray(outputs)
		outputs = outputs.reshape((-1, 128))
		outputs.shape

		# import sklearn
		# from sklearn.manifold import TSNE
		X_embedded = TSNE(2).fit_transform(outputs)
		X_embedded.shape

		# import numpy as np
		# from sklearn.decomposition import PCA
		X_PCA = PCA(3).fit_transform(outputs)
		print(X_PCA.shape)

		# X_embedded = TSNE(2).fit_transform(X_PCA)
		# print(X_embedded.shape)

		# import matplotlib.pyplot as plt
		color = 0
		for i in range(len((X_embedded))):
			el = X_embedded[i]
			if i % 51 == 0 and not i == 0:
				color += 1
				color = color % 10
			plt.scatter(el[0], el[1], color="C" + str(color))

		file1 = ('RGB-D_Face_database/faceid_train/(2012-05-16)(154211)/015_1_d.dat')
		inp1 = create_input_rgbd(file1)
		file1 = ('RGB-D_Face_database/faceid_train/(2012-05-16)(154211)/011_1_d.dat')
		inp2 = create_input_rgbd(file1)

		model_final.predict([inp1, inp2])

	#####################
	#
	# reverse
	#
	def reverse(self, data):
		for index in range(len(data) - 1, -1, -1):
			yield data[index]

	"""
	sprintfスタイル: '%s, %s' % ('Hello', 'World')
	拡張sprintfスタイル: '%(a)s, %(b)s' % dict(a='Hello', b='World')
	formatメソッド利用: '{0}, {1}'.format('Hello', 'World')
	"""




if __name__ == '__main__':
	train()

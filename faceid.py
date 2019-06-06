#! /usr/bin/env python
#!/usr/bin/env PYTHONIOENCODING=utf-8
# encoding: utf-8
# -*- coding: utf-8 -*-
# vim:fileencoding=UTF-8

######################
##
## M.Nakazawa
##

#
# Standard library imports.
#
from __future__ import absolute_import, print_function, unicode_literals

import os
import os.path
import subprocess
import sys
import argparse
import glob
import argparse
# import lzma
import tempfile
import shutil
from functools import partial
from io import StringIO

#
# Related third party imports.
#
import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from keras.models import Sequential, Model
# from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, concatenate
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate
from keras.layers import GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.activations import relu, softmax
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras import models
from keras.utils import plot_model
import keras.callbacks
import keras.backend.tensorflow_backend as KTF

import tensorflow as tf
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#
# toolz
#
from toolz import juxt, identity
from operator import add, mul

#
# funcy
#
#from funcy.py3 import whatever, you, need

#
# Local application/library specific imports.
#
from hyperdash import Experiment

#
# Local module
#
#	from . import filename

FS_ENCODING = sys.getfilesystemencoding()

#####################
#
# functions
#
def euclidean_distance(inputs):
	assert len(inputs) == 2, 'Euclidean distance needs 2 inputs, %d given' % len(inputs)
	u, v = inputs
	return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))


#####################
#
# contrastive loss
#
def contrastive_loss(y_true, y_pred):
	margin = 1.
	return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))


def linear(inputs):
	# assert len(inputs) == 2, 'Linear distance needs 2 inputs, %d given' % len(inputs)
	# u, v = inputs
	# return K.sqrt(K.sum((K.square(u-v)), axis=1, keepdims=True))
	return (inputs)


#####################
#
# custom activation
#
def custom_activation(x):
	return (K.sigmoid(x) * 5) - 1


#####################
#
# class Scaling
#
class Scaling(keras.layers.Layer):
	"""回帰で出力をスケーリングするために作った適当レイヤー"""

	def __init__(self, mean, std, **kwargs):
		self.mean = float(mean)
		self.std = float(std)
		assert self.std > 0
		super().__init__(**kwargs)

	def build(self, input_shape):
		self.W = self.add_weight(name='W', shape=(1,), initializer=keras.initializers.Constant(self.std), trainable=True)
		super().build(input_shape)

	def call(self, x, mask=None):
		return x * self.W + self.mean

	def get_config(self):
		config = {'mean': float(self.mean), 'std': float(self.std)}
		base_config = super().get_config()
		return dict(list(base_config.items()) + list(config.items()))


#####################
#
# CLASS
# Input preprocessing.
# Here we create some functions that will create the input couple for our model, both correct and wrong couples.
# I created functions to have both depth-only input and RGBD inputs.
#
#####################
class faceid_t:
	#
	# class variable (shared by all instances)
	#
	# __attr	= 100
	# val		= 10

	######################
	#
	# ctor
	#
	def __init__(self):
		print("face_id_t::init...")

	# pass
	#
	# instance variable (unique to each instance)
	#
	# self.__attr = 99
	# self.data	= data
	# self.index	= len(data)

	######################
	#
	# dtor
	#
	# def __del__(self):

	######################
	#
	# iter, next
	#
	# def __iter__(self):
	#   return( self )

	# def __next__(self):
	# 	if self.index == 0:
	# 		raise StopIteration
	# 	self.index = self.index - 1
	# 	return( self.data[self.index] )

	######################
	#
	# reverse
	#
	# def __repr__(self):
	# def __str__(self):
	# def __bytes__(self):
	# def __format__(self):

	######################
	#
	# text/print summary
	#
	def summary_text(self, model):
		text = ""
		with StringIO() as buf:
			model.summary(print_fn=lambda x: buf.write(x + "\n"))
			text = buf.getvalue()
		return text

	def summary_file(self, file, model):
		with open(file, "w") as fp:
			model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

	######################
	#
	# save/load model
	#
	"""
	def save_model(self, model, path):
		directory = os.path.dirname(path)
		with tempfile.NamedTemporaryFile(dir=directory) as f, lzma.open(path, 'wb') as wf:
			models.save_model(model, f.name)
			f.seek(0)
			shutil.copyfileobj(f, wf)

		model.save = partial(save_model, model)
		model.save('model.h5.xz')

	def load_model(self, path, custom_objects=None):
		with lzma.open(path, 'rb') as f, tempfile.NamedTemporaryFile(dir=directory) as wf:
			shutil.copyfileobj(f, wf)
			wf.seek(0)
			return models.load_model(wf.name, custom_objects=custom_objects)
	"""

	#####################
	#
	# create couple
	#
	def create_couple(self, file_path):
		folder = np.random.choice(glob.glob(file_path + "*"))
		while folder == "datalab":
			folder = np.random.choice(glob.glob(file_path + "*"))
		#
		# debug
		#
		# print(folder)
		# print( f"create_couple: file_path={file_path}, folder={folder}" )

		mat = np.zeros((480, 640), dtype='float32')
		i = 0
		j = 0
		depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
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
		mat_small = mat[140:340, 220:420]
		mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)
		#
		# debug
		#
		# plt.imshow(mat_small)
		# plt.show()

		mat2 = np.zeros((480, 640), dtype='float32')
		i = 0
		j = 0
		depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
		with open(depth_file) as file:
			for line in file:
				vals = line.split('\t')
				for val in vals:
					if val == "\n": continue
					if int(val) > 1200 or int(val) == -1: val = 1200
					mat2[i][j] = float(int(val))
					j += 1
					j = j % 640
				i += 1
			mat2 = np.asarray(mat2)
		mat2_small = mat2[140:340, 220:420]
		mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)
		#
		# debug
		#
		# plt.imshow(mat2_small)
		# plt.show()
		return np.array([mat_small, mat2_small])

	#####################
	#
	# create couple rgbd
	#
	def create_couple_rgbd(self, file_path):
		folder = np.random.choice(glob.glob(file_path + "*"))
		while folder == "datalab":
			folder = np.random.choice(glob.glob(file_path + "*"))
		#
		# debug
		#
		print(folder)
		print( f"create_couple_rgbd: file_path={file_path}, folder={folder}" )

		mat = np.zeros((480, 640), dtype='float32')
		i = 0
		j = 0
		depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
		print( f"create_couple_rgbd: 1 depth_file={depth_file}" )

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
			print( f"create_couple_rgbd: 1 mat.shape={mat.shape}" )

		mat_small = mat[140:340, 220:420]
		print( f"create_couple_rgbd: 1 mat_small.shape={mat_small.shape}" )
		img = Image.open(depth_file[:-5] + "c.bmp")
		img.thumbnail((640, 480))
		img = np.asarray(img)
		img = img[140:340, 220:420]
		#
		# debug
		#
		# plt.imshow( img )
		# plt.show()
		mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)
		# plt.imshow( mat_small )
		# plt.show()

		mat2 = np.zeros((480, 640), dtype='float32')
		i = 0
		j = 0
		depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
		print( f"create_couple_rgbd: 2 depth_file={depth_file}" )
		with open(depth_file) as file:
			for line in file:
				vals = line.split('\t')
				for val in vals:
					if val == "\n": continue
					if int(val) > 1200 or int(val) == -1: val = 1200
					mat2[i][j] = float(int(val))
					j += 1
					j = j % 640
				i += 1
			mat2 = np.asarray(mat2)
			print( f"create_couple_rgbd: 2 mat.shape={mat.shape}" )
		mat2_small = mat2[140:340, 220:420]
		print( f"create_couple_rgbd: 2 mat_small.shape={mat2_small.shape}" )
		img2 = Image.open(depth_file[:-5] + "c.bmp")
		img2.thumbnail((640, 480))
		img2 = np.asarray(img2)
		img2 = img2[160:360, 240:440]

		#
		# debug
		#
		# plt.imshow(img2)
		# plt.show()
		mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)
		# plt.imshow(mat2_small)
		# plt.show()

		full1 = np.zeros((200, 200, 4))
		full1[:, :, :3] = img[:, :, :3]
		full1[:, :, 3] = mat_small
		print( f"create_couple_rgbd: full1.shape={full1.shape}" )

		full2 = np.zeros((200, 200, 4))
		full2[:, :, :3] = img2[:, :, :3]
		full2[:, :, 3] = mat2_small
		print( f"create_couple_rgbd: full2.shape={full2.shape}" )
		return np.array([full1, full2])

	"""
	"""

	#####################
	#
	# create wrong
	#
	def create_wrong(self, file_path):
		folder = np.random.choice(glob.glob(file_path + "*"))
		while folder == "datalab":
			folder = np.random.choice(glob.glob(file_path + "*"))
		#
		# debug
		#
		# print( f"create_wrong: file_path={file_path}, folder={folder}" )

		mat = np.zeros((480, 640), dtype='float32')
		i = 0
		j = 0
		depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
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
		mat_small = mat[140:340, 220:420]
		mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)
		#
		# debug
		#
		# plt.imshow(mat_small)
		# plt.show()

		folder2 = np.random.choice(glob.glob(file_path + "*"))
		while folder == folder2 or folder2 == "datalab":  # it activates if it chose the same folder
			folder2 = np.random.choice(glob.glob(file_path + "*"))
		mat2 = np.zeros((480, 640), dtype='float32')
		i = 0
		j = 0
		depth_file = np.random.choice(glob.glob(folder2 + "/*.dat"))
		with open(depth_file) as file:
			for line in file:
				vals = line.split('\t')
				for val in vals:
					if val == "\n": continue
					if int(val) > 1200 or int(val) == -1: val = 1200
					mat2[i][j] = float(int(val))
					j += 1
					j = j % 640
				i += 1
			mat2 = np.asarray(mat2)
		mat2_small = mat2[140:340, 220:420]
		mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)
		#  plt.imshow(mat2_small)
		#  plt.show()
		return np.array([mat_small, mat2_small])

	"""
	"""

	#####################
	#
	# create wrong rgbd
	#
	def create_wrong_rgbd(self, file_path):
		folder = np.random.choice(glob.glob(file_path + "*"))
		while folder == "datalab":
			folder = np.random.choice(glob.glob(file_path + "*"))
		#
		# debug
		#
		# print( f"create_wrong_rgbd: file_path={file_path}, folder={folder}" )
		mat = np.zeros((480, 640), dtype='float32')
		i = 0
		j = 0
		depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
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
		mat_small = mat[140:340, 220:420]
		img = Image.open(depth_file[:-5] + "c.bmp")
		img.thumbnail((640, 480))
		img = np.asarray(img)
		img = img[140:340, 220:420]
		mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)
		#
		# debug
		#
		# plt.imshow( img )
		# plt.show()
		# plt.imshow( mat_small )
		# plt.show()

		folder2 = np.random.choice(glob.glob(file_path + "*"))
		while folder == folder2 or folder2 == "datalab":  # it activates if it chose the same folder
			folder2 = np.random.choice(glob.glob(file_path + "*"))
		mat2 = np.zeros((480, 640), dtype='float32')
		i = 0
		j = 0
		depth_file = np.random.choice(glob.glob(folder2 + "/*.dat"))
		with open(depth_file) as file:
			for line in file:
				vals = line.split('\t')
				for val in vals:
					if val == "\n": continue
					if int(val) > 1200 or int(val) == -1: val = 1200
					mat2[i][j] = float(int(val))
					j += 1
					j = j % 640
				i += 1
			mat2 = np.asarray(mat2)
		mat2_small = mat2[140:340, 220:420]
		img2 = Image.open(depth_file[:-5] + "c.bmp")
		img2.thumbnail((640, 480))
		img2 = np.asarray(img2)
		img2 = img2[140:340, 220:420]
		mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)
		# plt.imshow(img2)
		# plt.show()
		# plt.imshow(mat2_small)
		# plt.show()
		full1 = np.zeros((200, 200, 4))
		full1[:, :, :3] = img[:, :, :3]
		full1[:, :, 3] = mat_small

		full2 = np.zeros((200, 200, 4))
		full2[:, :, :3] = img2[:, :, :3]
		full2[:, :, 3] = mat2_small
		return np.array([full1, full2])

	#####################
	#
	# euclidean distance
	#
	def euclidean_distance(self, inputs):
		assert len(inputs) == 2, 'Euclidean distance needs 2 inputs, %d given' % len(inputs)
		u, v = inputs
		return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))

	#####################
	#
	# contrastive loss
	#
	def contrastive_loss(self, y_true, y_pred):
		margin = 1.
		return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

	#####################
	#
	# fire
	#
	def fire(self, x, squeeze=16, expand=64):
		x = Convolution2D(squeeze, (1, 1), padding='valid')(x)
		x = Activation('relu')(x)

		left = Convolution2D(expand, (1, 1), padding='valid')(x)
		left = Activation('relu')(left)

		right = Convolution2D(expand, (3, 3), padding='same')(x)
		right = Activation('relu')(right)

		x = concatenate([left, right], axis=3)
		return x

	#####################
	#
	# generator
	#
	def generator(self, batch_size):
		while 1:
			x = []
			y = []
			# print("generator0:")
			# print(x)
			# print(y)
			switch = True
			for _ in range(batch_size):
				if switch:
					x.append(self.create_couple_rgbd("RGB-D_Face_database/faceid_train/").reshape((2, 200, 200, 4)))
					y.append(np.array([0.]))
				else:
					x.append(self.create_wrong_rgbd("RGB-D_Face_database/faceid_train/").reshape((2, 200, 200, 4)))
					y.append(np.array([1.]))
				switch = not switch
			x = np.asarray(x)
			y = np.asarray(y)
			# XX1 = x[0, :]
			# XX2 = x[1, :]
			# print("generator:")
			# print(x)
			# print(y)
			yield [x[:, 0], x[:, 1]], y

	#####################
	#
	# val generator
	#
	def val_generator(self, batch_size):
		while 1:
			x = []
			y = []
			# print("val_generator0:")
			# print(x)
			# print(y)
			switch = True
			for _ in range(batch_size):
				if switch:
					x.append(self.create_couple_rgbd("RGB-D_Face_database/faceid_val/").reshape((2, 200, 200, 4)))
					y.append(np.array([0.]))
				else:
					x.append(self.create_wrong_rgbd("RGB-D_Face_database/faceid_val/").reshape((2, 200, 200, 4)))
					y.append(np.array([1.]))
				switch = not switch
			x = np.asarray(x)
			y = np.asarray(y)
			# XX1 = x[0, :]
			# XX2 = x[1, :]
			# print("val_generator:")
			# print(x)
			# print(y)
			yield [x[:, 0], x[:, 1]], y

	#####################
	#
	# run
	#
	def run(self):
		#
		#	hyperdash
		#
		exp = Experiment("faceid")
		hd_callback = Hyperdash(exp=exp)

		# print(self.create_couple("RGB-D_Face_database/faceid_train/"))
		# print(self.create_couple_rgbd("RGB-D_Face_database/faceid_val/"))
		self.create_couple_rgbd("RGB-D_Face_database/faceid_val/")
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

		x = self.fire(x, squeeze=16, expand=16)
		x = self.fire(x, squeeze=16, expand=16)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

		x = self.fire(x, squeeze=32, expand=32)
		x = self.fire(x, squeeze=32, expand=32)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

		x = self.fire(x, squeeze=48, expand=48)
		x = self.fire(x, squeeze=48, expand=48)
		x = self.fire(x, squeeze=64, expand=64)
		x = self.fire(x, squeeze=64, expand=64)
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
		lambda_merge = Lambda(self.euclidean_distance)([feat_x1, feat_x2])

		model_final = Model(inputs=[im_in1, im_in2], outputs=lambda_merge)
		print("\nmodel final summary")
		model_final.summary()
		plot_model(model_final, show_shapes=True, to_file='model_final.png')

		adam = Adam(lr=0.001)
		sgd = SGD(lr=0.001, momentum=0.9)

		model_final.compile(optimizer=adam, loss=self.contrastive_loss)

		#
		# plot model
		#
		# print("write model summary png...")
		# plot_model(model_final, show_shapes=True, to_file='model.png')
		# print("write model summary png...done")

		#
		# generator
		#
		gen = self.generator(16)
		val_gen = self.val_generator(4)

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
		# generator
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
			gen,
			steps_per_epoch=10,		# 30
			epochs=1,				# 50
			verbose=1,
			# callbacks=[checkpointer],
			# callbacks=[checkpointer, hd_callback],
			callbacks=[checkpointer, csv_logger, early_stop, reduce_lr, hd_callback],
			# callbacks=[checkpointer, pbarl, csv_logger, early_stop, reduce_lr, hd_callback],
			# callbacks=[checkpointer, csv_logger, early_stop, reduce_lr, tensorboard, hd_callback],
			# pickle_safe=True,
			validation_data=val_gen,
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
		cop = self.create_couple("RGB-D_Face_database/faceid_val/")
		score = model_final.evaluate([cop[0].reshape((1, 200, 200, 4)), cop[1].reshape((1, 200, 200, 4))], np.array([0.]))
		print('Test score(couple):', score[0])
		print('Test accuracy(couple):', score[1])

		cop = self.create_wrong_rgbd("RGB-D_Face_database/faceid_val/")
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
		exp.end()

	#####################
	#
	# upload
	#
	"""
	def upload(self):
		# モデルの読込(再 model.compile() は不要)
		#model = loadl_model("RGB-D_Face_database/snapshot/model_20180815a.h5")

		from google.colab import files

		# Install the PyDrive wrapper & import libraries.
		# This only needs to be done once in a notebook.
		#	!pip install -U -q PyDrive
		from pydrive.auth import GoogleAuth
		from pydrive.drive import GoogleDrive
		from google.colab import auth
		from oauth2client.client import GoogleCredentials

		# Authenticate and create the PyDrive client.
		# This only needs to be done once in a notebook.
		auth.authenticate_user()
		gauth = GoogleAuth()
		gauth.credentials = GoogleCredentials.get_application_default()
		drive = GoogleDrive(gauth)

		# Create & upload a file.
		uploaded = drive.CreateFile({'title': 'faceid_big_rgbd.h5'})
		uploaded.SetContentFile('faceid_big_rgbd.h5')
		uploaded.Upload()
		print('Uploaded file with ID {}'.format(uploaded.get('id')))
	"""

	#####################
	#
	# download
	#
	"""
	def download(self):
		# Install the PyDrive wrapper & import libraries.
		# This only needs to be done once per notebook.
		#	!pip install -U -q PyDrive
		from pydrive.auth import GoogleAuth
		from pydrive.drive import GoogleDrive
		from google.colab import auth
		from oauth2client.client import GoogleCredentials

		# Authenticate and create the PyDrive client.
		# This only needs to be done once per notebook.
		auth.authenticate_user()
		gauth = GoogleAuth()
		gauth.credentials = GoogleCredentials.get_application_default()
		drive = GoogleDrive(gauth)

		# Download a file based on its file ID.
		#
		# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
		file_id = '17Lo_ZxYcKO751iYs4XRyIvVXME8Lyc75'
		downloaded = drive.CreateFile({'id': file_id})
		#print('Downloaded content "{}"'.format(downloaded.GetContentString()))

		downloaded.GetContentFile('pesi.h5')
	"""

	#####################
	#
	# download
	#
	def save_model(model, options):
		json_string = model.to_json()
		open(options['file_arch'], 'w').write(json_string)
		model.save_weights(options['file_weight'])

	def load_model(options):
		model = model_from_json(open(options['file_arch']).read())
		model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
		model.load_weights(options['file_weight'])
		return model

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
		#  print(folder)
		mat = np.zeros((480, 640), dtype='float32')
		i = 0
		j = 0
		depth_file = file_path
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
		mat_small = mat[140:340, 220:420]
		img = Image.open(depth_file[:-5] + "c.bmp")
		img.thumbnail((640, 480))
		img = np.asarray(img)
		img = img[140:340, 220:420]
		mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)

		# plt.figure(figsize=(8,8))
		# plt.grid(True)
		# plt.xticks([])
		# plt.yticks([])
		# plt.imshow(mat_small)
		# plt.show()
		# plt.figure(figsize=(8,8))
		# plt.grid(True)
		# plt.xticks([])
		# plt.yticks([])
		# plt.imshow(img)
		# plt.show()

		full1 = np.zeros((200, 200, 4))
		full1[:, :, :3] = img[:, :, :3]
		full1[:, :, 3] = mat_small

		return np.array([full1])

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


#####################
#
#	main
#
def main(argv=sys.argv[1:]):
	#
	#
	# argparse.ArgumentParser::add_argument
	# name または flags - 名前か、あるいはオプション文字列のリスト (例: foo や -f, --foo)。
	# action - コマンドラインにこの引数があったときのアクション。
	# nargs - 受け取るべきコマンドライン引数の数。
	# const - 一部の action と nargs の組み合わせで利用される定数。
	# default - コマンドラインに引数がなかった場合に生成される値。
	# type - コマンドライン引数が変換されるべき型。
	# choices - 引数として許される値のコンテナー。
	# required - コマンドラインオプションが省略可能かどうか (オプション引数のみ)。
	# help - 引数が何なのかを示す簡潔な説明。
	# metavar - 使用法メッセージの中で使われる引数の名前。
	# dest - parse_args() が返すオブジェクトに追加される属性名。

	# parser	= argparse.ArgumentParser()
	# parser.add_argument( 'output', type=argparse.FileType('w') )
	# parser.add_argument( '-f', '--format', default='json', choices=['json', 'csv'] )
	# args	= parser.parse_args(argv)
	# fmt		= args.format
	# output	= args.output
	# if fmt == 'json':
	# elif fmt == 'csv':
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False, help='If true, uses fake data for unit testing.')
	parser.add_argument('--max_steps', type=int, default=1000, help='Number of steps to run trainer.')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
	parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout.')
	parser.add_argument('--data_dir', type=str, default='/tmp/data', help='Directory for storing data')
	parser.add_argument('--summaries_dir', type=str, default='/tmp/mnist_logs', help='Summaries directory')
	FLAGS = parser.parse_args()
	if train or FLAGS.fake_data:
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--train', action="store_true", help="trainning processed")
	parser.add_argument('-p', '--predict', action="store_true", help="prediction execute")
	parser.add_argument('-m', "--model", help="pretrained NN model")
	args = parser.parse_args()

	faceid = faceid_t()

	if args.model:
		pass

	elif args.train:
		print('training...')
		faceid.run()
		print('training...done')

	elif args.predict:
		print('predict...')
		faceid.predict()
		print('predict...done')

	else:
		print('predict...')
		faceid.predict()
		print('predict...done')

	"""
	argc = len(argv)
	if (argc == 1) and (argv[0]=="train"):
		print( 'training...' )
		faceid.run()

	elif (argc == 1) and (argv[0]=="predict"):
		print( 'predict...' )
		faceid.predict()

	else:
		print( 'you must set \"train\" or \"predict\"...' )
	"""

	print('all...done')


if __name__ == '__main__':
	sys.exit(main())

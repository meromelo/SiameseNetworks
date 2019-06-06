#!/usr/bin/env python
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
# Related third party imports.
#
#
#   plaidml
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
from keras.layers.convolutional import Convolution2D
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
from funcy						import concat, identity, juxt, partial, rcompose, repeat, take

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


FS_ENCODING = sys.getfilesystemencoding()

#####################
#
# SqueezeNet
# https://tail-island.github.io/programming/2017/10/25/keras-and-fp.html
#
# SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
# https://arxiv.org/abs/1602.07360
#
class SqueezeNet():
	#
	# class variables
	#

	#####################
	#
	# ctor
	#
	def	__init__(self):
		pass

	#####################
	#
	# return list by juxt in funcy
	#
	def	__ljuxt(self, *args):
		return rcompose(juxt(*args), list)

	#####################
	#
	# batch normalization
	#
	def	__batch_normalization(self):
		return BatchNormalization()

	#####################
	#
	# relu
	#
	def	__relu(self):
		return Activation('relu')

	#####################
	#
	# conv
	# ReLUするならウェイトをHe初期化するのが基本らしい。
	# Kerasにはweight decayがなかったのでkernel_regularizerで代替したのたけど、これで正しい？
	#
	def	__conv(self, filters, kernel_size):
		return Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))

	#####################
	#
	# concatenate
	#
	def	__concatenate(self):
		return Concatenate()

	#####################
	#
	# add
	#
	def	__add(self):
		return Add()

	#####################
	#
	# max pooling
	#
	def	__max_pooling(self):
		return MaxPooling2D()

	#####################
	#
	# dropout
	#
	def	__dropout(self):
		return Dropout(0.5)

	#####################
	#
	# global averate pooling
	#
	def	__global_average_pooling(self):
		return GlobalAveragePooling2D()

	#####################
	#
	# dense
	#
	def	__dense(self, units, activation, a=0.0001):
		return Dense(units, activation=activation, kernel_regularizer=l2(a))

	#####################
	#
	# softmax
	#
	def	__softmax(self):
		return Activation('softmax')

	#####################
	#
	# define SqueezeNet
	#
	def	__fire_module(self, filters_squeeze, filters_expand):
		return rcompose(
				self.__batch_normalization(),
				self.__relu(),
				self.__conv(filters_squeeze, 1),
				self.__batch_normalization(),
				self.__relu(),
				self.__ljuxt(
					self.__conv(filters_expand // 2, 1),
					self.__conv(filters_expand // 2, 3)
					),
				self.__concatenate()
				)

	#####################
	#
	# define SqueezeNet (shortcut)
	#
	def	__fire_module_with_shortcut(self, filters_squeeze, filters_expand):
		return rcompose(
				self.__ljuxt(
					self.__fire_module(filters_squeeze, filters_expand),
					identity
					),
				self.__add()
				)

	#####################
	#
	# make SqueeezeNet graph
	#
	def	make_graph(self, class_size):
		return rcompose(
				self.__conv(96, 3),
				# self.__max_pooling(),
				self.__fire_module(16, 128),
				self.__fire_module_with_shortcut(16, 128),
				self.__fire_module(32, 256),
				self.__max_pooling(),
				self.__fire_module_with_shortcut(32, 256),
				self.__fire_module(48, 384),
				self.__fire_module_with_shortcut(48, 384),
				self.__fire_module(64, 512),
				self.__max_pooling(),
				self.__fire_module_with_shortcut(64, 512),
				self.__batch_normalization(),
				self.__relu(),
				# self.__conv(class_size, 1),
				self.__global_average_pooling(),
				# self.__softmax(),
				self.__dense(class_size, 'softmax')
			)



#####################
#
# Wide ResNet
# https://tail-island.github.io/programming/2017/10/25/keras-and-fp.html
#
# Identity Mappings in Deep Residual Networks
# https://arxiv.org/abs/1603.05027)# Wide Residual Network
#
class WideResNet():
	#
	# class variables
	#

	#####################
	#
	# ctor
	#
	def	__init__(self):
		pass

	#####################
	#
	# return list by juxt in funcy
	#
	def	__ljuxt(self, *fs):
		return rcompose(juxt(*fs), list)

	#####################
	#
	# batch normalization
	#
	def	__batch_normalization(self):
		return BatchNormalization()

	#####################
	#
	# relu
	#
	def	__relu(self):
		return Activation('relu')

	#####################
	#
	# conv
	# ReLUしたいならウェイトをHe初期化するのが基本らしい。
	# Kerasにはweight decayがないので、kernel_regularizerで代替しました。
	#
	def	__conv(self, filter_size, kernel_size, stride_size=1):
		return Conv2D(
				filter_size,
				kernel_size,
				strides=stride_size,
				padding='same',
				kernel_initializer='he_normal',
				kernel_regularizer=l2(0.0005),
				use_bias=False
				)

	#####################
	#
	# add
	#
	def	__add(self):
		return Add()

	#####################
	#
	# global average pooling
	#
	def	__global_average_pooling(self):
		return GlobalAveragePooling2D()

	#####################
	#
	# dense
	# Kerasにはweight decayがないので、kernel_regularizerで代替しました。
	#
	def	__dense(self, units, activation, a=0.0005):
		return Dense( units, activation=activation, kernel_regularizer=l2(a))

	#####################
	#
	# first residual unit
	# Define WRN-28-10
	#
	def	__first_residual_unit(self, filter_size, stride_size):
		return rcompose(
				self.__batch_normalization(),
				self.__relu(),
				self.__ljuxt(
					rcompose(
						self.__conv(filter_size, 3, stride_size),
						self.__batch_normalization(),
						self.__relu(),
						self.__conv(filter_size, 3, 1)
						),
					rcompose(self.__conv(filter_size, 1, stride_size))
					),
				self.__add()
				)

	#####################
	#
	# residual_unit
	#
	def	__residual_unit(self, filter_size):
		return rcompose(
				self.__ljuxt(
					rcompose(
						self.__batch_normalization(),
						self.__relu(),
						self.__conv(filter_size, 3),
						self.__batch_normalization(),
						self.__relu(),
						self.__conv(filter_size, 3)
						),
					identity
					),
				self.__add()
				)

	#####################
	#
	# residual_block
	#
	def	__residual_block(self, filter_size, stride_size, unit_size):
		return rcompose(
				self.__first_residual_unit(filter_size, stride_size),
				rcompose(
					*repeatedly(
						partial(__residual_unit, filter_size),
						unit_size-1
						)
					)
				)

	#####################
	#
	# ctor
	#
	#
	# k = 10  # 論文によれば、CIFAR-10に最適な値は10。
	# n =  4  # 論文によれば、CIFAR-10に最適な値は4。
	# WRN-28なのに4になっているのは、28はdepthで、
	# depthはconvの数で、1（最初のconv）+ 3 * n * 2 + 3（ショートカットのconv？）だからみたい。
	#
	def	make_graph(self, class_size):
		return rcompose(
				self.__conv(16, 3),
				self.__residual_block(16 * k, 1, n),
				self.__residual_block(32 * k, 2, n),
				self.__residual_block(64 * k, 2, n),
				self.__batch_normalization(),
				self.__relu(),
				self.__global_average_pooling(),
				self.__dense(class_size, 'softmax')
				)



#####################
#
# SiameseNet
# http://ni4muraano.hatenablog.com/entry/2019/01/06/145827
#
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda, Conv2D, Activation, MaxPool2D, BatchNormalization, Dropout, Flatten
import keras.backend as K

class SiameseNet(object):
	def __init__(self, input_shape, feature_dim):
		seq = Sequential()
		seq.add(Conv2D(16, 3, padding='same', input_shape=input_shape))
		seq.add(BatchNormalization())
		seq.add(Activation('relu'))
		seq.add(MaxPool2D())

		seq.add(Conv2D(32, 3, padding='same'))
		seq.add(BatchNormalization())
		seq.add(Activation('relu'))
		seq.add(MaxPool2D())

		seq.add(Conv2D(64, 3, padding='same'))
		seq.add(BatchNormalization())
		seq.add(Activation('relu'))
		seq.add(MaxPool2D())

		seq.add(Flatten())
		seq.add(Dense(256, activation='sigmoid'))
		seq.add(Dropout(0.2))
		seq.add(Dense(feature_dim, activation='linear'))

		input_a = Input(shape=input_shape)
		input_b = Input(shape=input_shape)
		processed_a = seq(input_a)
		processed_b = seq(input_b)
		distance = Lambda(self._euclidean_distance, output_shape=self._eucl_dist_output_shape)([processed_a, processed_b])
		self._model = Model(inputs=[input_a, input_b], outputs=distance)

	def _euclidean_distance(self, vects):
		x, y = vects
		distance = K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
		return distance

	def _eucl_dist_output_shape(self, shapes):
		shape1, shape2 = shapes
		return (shape1[0], 1)

	def get_model(self):
		return self._model

	def contrastive_loss(y_true, y_pred):
		'''Contrastive loss from Hadsell-et-al.'06
		http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
		'''
		margin = 1
		return K.mean(y_true*K.square(y_pred) + (1 - y_true)*K.square(K.maximum(margin - y_pred, 0)))









def	main():

	#
	# CIFAR-10
	#
	cifar = CIFAR_10()

	#
	# x_train.shape		= (50000, 32, 32, 3)
	# y_train.shape		= (50000, 10)
	# x_validation.shape= (10000, 32, 32, 3)
	# y_validation.shape= (10000, 10)
	#
	data			= cifar.load_data()
	x_train			= data['training_data']
	y_train			= data['training_label']
	x_validation	= data['validation_data']
	y_validation	= data['validation_label']
	print("x_train.shape=", x_train.shape)
	print("y_train.shape=", y_train.shape)
	print("x_validation.shape=", x_validation.shape)
	print("y_validation.shape=", y_validation.shape)


	#
	# SqueezeNet
	#
	squeeze = SqueezeNet()
	i = Input(shape=x_train.shape[1:])
	o = squeeze.make_graph(y_train.shape[1])(i)

	#
	# model
	#
	model = Model(inputs=i, outputs=o)

	#
	# compile model
	#
	model.compile(
			loss='categorical_crossentropy',
			optimizer=SGD(momentum=0.9),
			metrics=['accuracy']
			)

	#
	# generator in ImageDataGenerator by keras
	#
	train_data = ImageDataGenerator(
			featurewise_center=True,
			featurewise_std_normalization=True,
			width_shift_range=0.125,
			height_shift_range=0.125,
			horizontal_flip=True
			)
	validation_data = ImageDataGenerator(
			featurewise_center=True,
			featurewise_std_normalization=True
			)
	for data in (train_data, validation_data):
		data.fit(x_train)  # 実用を考えると、x_validationでのfeaturewiseのfitは無理だと思う… … 。

	#
	# check pickle
	#
	# file_pickle = "./results/history.pickle"
	model_path		= "./results"
	model_file  	= model_path + "/model.h5"
	model_weights	= model_path + "/weights.h5"
	print(f"models: model={model_file}, weight={model_weights}" )
	# print(f"models: arch  =", options['file_arch'])
	# print(f"models: weight=", options['model_weights'])
	if not path.exists(model_path):
		os.mkdir(model_path)

	#
	# print model
	#
	from lib_utils import print_model_summary
	print_model_summary(model, "./results/network.txt", "model.png")


	#
	# check model, if not exist trained model, we have to make trained parameters for model.
	#
	if not path.exists(model_file):

		#
		# fit generator
		#
		batch_size = 1000	# 100
		epochs     = 1		# 200
		results = model.fit_generator(
			#
			# generate train data (ImageDataGenerator by keras)
			#
			train_data.flow(x_train, y_train, batch_size=batch_size),

			#
			# steps/epoch
			#
			steps_per_epoch=x_train.shape[0] // batch_size,

			#
			# epoch
			#
			epochs=epochs,

			#
			# callbacks
			#
			callbacks = [
				LearningRateScheduler(
					partial(
						getitem,
						tuple(take(epochs, concat(repeat(0.010, 1), repeat(0.100, 99), repeat(0.010, 50), repeat(0.001))))
						)
					)
				],
			#
			# generate validation data (ImageDataGenerator by keras)
			#
			validation_data=validation_data.flow(x_validation, y_validation, batch_size=batch_size),

			#
			# validation step
			#
			validation_steps=x_validation.shape[0] // batch_size,

			#
			# max_queue_size
			#
			max_queue_size=4
			)

		#
		# save keras model
		#
		from lib_utils import save_model_by_keras
		save_model_by_keras(model, model_file, model_weights)

		# del model

	else:
		#
		# load keras model
		#
		if path.exists(model_file):
			print("load model...")
			from lib_utils import load_model_by_keras
			model = load_model_by_keras(model_file, model_weights)
			print("load model...done")
		else:
			print("load model...: not found=", model_file, model_weights )

	#
	# check version
	#
	from lib_utils import get_version
	get_version(model_file)

		
	#
	# evaluate
	#
	"""
	print("model evaluate...")
	score = lmodel.evaluate(x_validation, y_validation, verbose=1)
	print("model evaluate: loss=", score[0])
	print("model evaluate: accuracy=", score[1])
	"""

	#
	# prediction
	#
	print("model prediction...")
	# lmodel.predict(y_validation.shape[1])
	# lmodel.predict(x_train.shape[1:])
	print("x_validation.shape=", x_validation.shape)
	print("x_validation.shape[0]=", x_validation.shape[0])
	print("x_validation.shape[1]=", x_validation.shape[1])
	print("x_validation.shape[2]=", x_validation.shape[2])
	print("x_validation.shape[3]=", x_validation.shape[3])
	i0 = x_validation[0:1]
	i1 = x_validation.reshape(10000,32,32,3)
	i2 = i1[0]
	print("i0.shape=", i0.shape)
	print("i1.shape=", i1.shape)
	print("i2.shape=", i2.shape)
	# lmodel.predict(i0, verbose=1)
	predo = model.predict(x_validation, verbose=1)[0]
	print(predo)

	"""
	"""
	preds = model.predict(x_validation, verbose=1)

	# for pre in preds:
	# 	y = pre.argmax()
	# 	print("label: ", y_validation[y])

	print('done')

if __name__ == '__main__':
	main()

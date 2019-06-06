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
#	snooper
#	https://blog.ikedaosushi.com/entry/2019/04/28/085904
#
#	import pysnooper
#
#	@pysnooper.snoop()
#	def XXX:
#	@pysnooper.snoop('/my/log/file.log')
#	@pysnooper.snoop(prefix='MyPrefix ')
#	@pysnooper.snoop(variables=('foo.bar',  'self.a'))
#

#
#	memo
#
#	1. Ternary conditionals
#		x = 1 if condition else 0
#
#	2. Underscore placeholders
#		num1 = 10_000_000_000
#		print(f'{num1:,}')
#
#	3. Context managers
#		with open('test.txt','r') as f:
#			file_contents = f.read()
#
#	4. Enumerate
#		for index, name in enumerate(names,start=1):
#			print(index, name)
#
#	5. Zip
#		names = ['Peter Parker','Clark Kent','Bruce Wayne']
#		heros = ['Spiderman','Superman','Batman']
#		for name, hero in zip (names, heros):
#			print(f'{name} is actually {helo}')
#
#		names = ['Peter Parker','Clark Kent','Bruce Wayne']
#		heros = ['Spiderman','Superman','Batman']
#		universes = ['Marvel','DC','DC']
#		for name, hero, universe in zip (names, heros, universes):
#			print(f'{name} is actually {helo} from {universe}')
#
#	6. Unpacking
#		a, b = (1, 2)
#		a, _ = (1,2)
#
#		x = (0, 1, 2, 3, 4)
#		a, b, *c = x
#
#	7. Setattr/Getattr
#		class Person():
#			pass
#		person = Person()
#
#		first_key = 'first'
#		first_val = 'Bun'
#
#		#setattrを使わない場合
#		person.first_key = first_val
#		print(person.first)
#
#		#setattrを使う場合
#		setattr(person, first_key, first_val)
#		#setattr(オブジェクト,属性名,属性値)
#		first = getattr(person, first_key)
#		print(first)
#
#
#		class Person():
#			pass
#		person = Person()
#
#		person_info = {'first':'Bun', 'last':'Tarou'}
#		for key, value in person_info.items():
#			setattr(person, key, value)
#
#		for key in person_info.keys():
#			print(getattr(person, key))
#
#	8. GetPass
#		from getpass import getpass
#		username = input('Yourname: ')
#		password = getpass('Password: ')
#		repassword = getPass('Again: ')
#		print('Logging in...')
#
#	9. format
#		print("{0:+6.2f} {1:s} {2:+9.2e}".format( 3.0, "a", 7.0 ), end="") )
#		'{0},  {1},  {2 }...'.format(変数1, 変数2, 変数3….)						# インデックス（添え字）で指定
#		'{h1}, {h2}, {h3}...'.format(h1=変数1, h2=変数2, h3=変数…)				# キーワード引数で指定
#		'{h1}, {h2}, {h3}...'.format(**{'h1':変数1, 'h2':変数2, 'h3':変数3….} )	# 辞書で指定
#
#		decimal = 106
#		print('{0}は2進数だと{0:b}、8進数だと{0:o}、16進数だと{0:X}'.format(decimal))
#		106は2進数だと1101010、8進数だと152、16進数だと6A
#
#		string1 = '左詰め'
#		string2 = '中央寄せ'
#		string3 = '右詰め'
#		print('{0:<10}'.format(string1))
#		print('{0:^10}'.format(string2))
#		print('{0:>10}'.format(string3))
#
#		animal = ('Dog', 'Cat')
#		name   = ('Maggie', 'Missy')
#		'I have a {0[0]} named {1[0]}'.format(animal,name)	#I have a dog named Maggie
#		'I have a {0[1]} named {1[1]}'.format(animal,name)	#I have a cat named Missy
#		'{:<50}'.format('aligned left')		#'aligned left (50インデックス用意して左詰めで文字入力)
#		'{:a<50}.format('aligned left ')	#'aligned left aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
#		'{:>50}'.format('aligned right')
#		'{:^50}'.format('aligned center')
#		'{:$^50}'.format('More Money')		#'$$$$$$$$$$$$$$$$$$$$More Money$$$$$$$$$$$$$$$$$$$$'
#		'Binary: {0:b}'.format(324)			#Binary: 101000100
#		'{:,}'.format(123456787654321)		#'123,456,787,654,321'
#		correct = 78
#		total = 84
#		'Your score is: {:.1%}'.format(correct/total)	#'Your score is: 92.9%'
#		'Your score is: {:.3%}'.format(correct/total)	#'Your score is: 92.857%'
#
#		f-strings
#		https://note.nkmk.me/python-f-strings/
#		print(f'zero padding: {i:08}')		# zero padding: 00001234
#		print(f'comma: {i:,}')				# comma: 1,234
#		print(f'bin: {i:b}')				# bin: 10011010010
#		print(f'oct: {i:o}')				# oct: 2322
#		print(f'hex: {i:x}')				# hex: 4d2
#		print(f'bin: {i:#b}')				# bin: 0b10011010010
#		print(f'oct: {i:#o}')				# oct: 0o2322
#		print(f'hex: {i:#x}')				# hex: 0x4d2
#		print(f'digit(decimal): {f:.3f}')	# digit(decimal): 12.346
#		print(f'digit(all)    : {f:.3g}')	# digit(all)    : 12.3
#		print(f'exponen: {f:.3e}')			# exponen: 1.235e+01
#		print(f'percent: {f:.2%}')			# percent: 12.30%
#		for i in range(5):
#			print(f'{f:.{i}f}')				# 1 1.2 1.23 1.234 1.2345
#		print('x\ty')						# x y
#		print(r'x\ty')						# x\ty
#
#		f'{number:0>10}'					# 右寄せ	#=> '0000000123'
#		f'{number:0^10}'					# 中央寄せ	#=> '0001230000'
#		f'{number:0<10}'					# 左寄せ	#=> '1230000000'
#
#		f'{number:{padding_char}>{padding_length}}'
#
#		https://note.nkmk.me/python-format-zero-hex/
#		https://gammasoft.jp/blog/python-string-format/
#		https://codom.hatenablog.com/entry/2016/12/27/000000
#
#
#
# Local module
#
#	from . import filename
#
#
#	A. argv
#		len(sys.argv)
#		print('sys.argv[0]      : ', sys.argv[0])
#		print('sys.argv[1]      : ', sys.argv[1])
#		print('sys.argv[2]      : ', sys.argv[2])
#
#	B. variable argv
#		def variable_args(*args, **kwargs):
#		print( 'args is', args )
#		print( 'kwargs is', kwargs )
#
#		variable_args('one', 'two', x=1, y=2, z=3)
#		args is ('one', 'two')
#		kwargs is {'y': 2, 'x': 1, 'z': 3}
#
#		可変長位置引数 (Define variable arguments)
#		def 会話(挨拶, *それ以降):
#	    print(挨拶)
#		for 言葉 in それ以降:
#			print(言葉)
#		>>> 会話('おはよう')
#		おはよう
#		>>> 会話('おはよう', '今日は天気いいね', 'え、なんで無視するの！？')
#		おはよう
#		今日は天気いいね
#		え、なんで無視するの！？
#
#		可変長キーワード引数 (Define keyword variable arguments)
#
#		def ラーメン(味, **細かい注文):
#			print(f'{味}ラーメン')
#			for key, value in 細かい注文.items():
#				print(f'{key}{value}')
#			print('ほか全部普通で！')
#		>>> ラーメン('醤油', 麺='硬め', 油='少なめ', チャシュー='あり', コーン='なし')
#		醤油ラーメン
#		麺硬め
#		油少なめ
#		チャシューあり
#		コーンなし
#		ほか全部普通で！
#


#
# Standard library imports.
#
from __future__	import absolute_import, print_function, unicode_literals

#
# Standard library imports.
#
import os
import os.path	as path
import subprocess
import sys
import argparse
import glob
import tempfile
import shutil
import lzma
from functools	import partial
from io			import StringIO

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
from keras.layers				import Add, Conv2D, Dense, MaxPooling2D, Activation, Flatten, Dropout, Lambda, ELU, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, concatenate
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling		import MaxPooling2D, AveragePooling2D
from keras.models				import Model, Sequential, save_model
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
import numpy				as np
import matplotlib.pyplot	as plt

#
# tensorflow
#
import tensorflow			as tf

#
# sklearn
#
import sklearn
from sklearn.manifold		import TSNE
from sklearn.decomposition	import PCA

#
# funcy
#
from funcy					import concat, identity, juxt, partial, rcompose, repeat, take

#
# hyperdash
#
from hyperdash				import Experiment as hyperdash_experiment
from lib_observe			import hyperdash_callback
#from hyperdash_callback		import Hyperdash

#
# Local application/library specific imports.
#
#from data_set					import CIFAR_10
from operator					import getitem


#
# encondig
# FS_ENCODING = sys.getfilesystemencoding()
#

#####################
#
# euclidean distance
#
def euclidean_distance(inputs:list)->float:
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
# linear
#
def linear(inputs:list)->list:
	# assert len(inputs) == 2, 'Linear distance needs 2 inputs, %d given' % len(inputs)
	# u, v = inputs
	# return K.sqrt(K.sum((K.square(u-v)), axis=1, keepdims=True))
	return (inputs)


#####################
#
# custom activation
#
def custom_activation(x:float)->float:
	return (K.sigmoid(x) * 5) - 1


######################
#
# print model summary (text/file)
#
def print_model_summary(model:keras.Model, file = "", image="")->str:
	#
	# file
	#
	if len(file) > 0:
		with open(file, "w") as fp:
			model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

	#
	# image
	#
	if len(image) > 0:
		keras.utils.plot_model(model, show_shapes=True, to_file=image)

	#
	# text
	#
	text = ""
	with StringIO() as buf:
		model.summary(print_fn=lambda x: buf.write(x + "\n"))
		text = buf.getvalue()

	return text


######################
#
# Plot training & validation accuracy values
#
def	plot_training_and_validation_accuracy_values(history:keras.history.history)->None:
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

######################
#
# save model with lzma (keras)
# https://kuune.org/text/2017/05/06/compress-and-save-keras-model/
#
def save_model_with_lzma(model:keras.Model, path:str, json:bool=True, yaml:bool=True, weights:bool=True)->None:
	directory = os.path.dirname(path)
	with tempfile.NamedTemporaryFile(dir=directory) as f, lzma.open(path, 'wb') as wf:
		keras.models.save_model(model, f.name)
		f.seek(0)
		shutil.copyfileobj(f, wf)

	model.save = partial(keras.models.save_model, model)
	model.save('model.h5.xz')

	#
	# save model (json, yaml, weights)
	#
	if (json):
		json_string = model.to_json()
		open(os.path.join(path, 'architecture.json'), 'w').write(json_string)

	if (yaml):
		yaml_string = model.to_yaml()
		open(os.path.join(path, 'architecture.yaml'), 'w').write(yaml_string)

	if (weights):
		model.save_weights(os.path.join(path, 'weights.hdf5'))


######################
#
# load model with lzma (keras)
# https://kuune.org/text/2017/05/06/compress-and-save-keras-model/
#
def load_model_with_lzma(path:str, custom_objects=None)->keras.Model:
	directory = os.path.dirname(path)
	with lzma.open(path, 'rb') as f, tempfile.NamedTemporaryFile(dir=directory) as wf:
		shutil.copyfileobj(f, wf)
		wf.seek(0)
		return keras.models.load_model(wf.name, custom_objects=custom_objects)


######################
#
# save model (keras)
# https://jovianlin.io/saving-loading-keras-models/
#
def	save_model_by_keras(model:keras.models.Model, model_file:str="model.h5", weights_file:str="weights.h5")->None:
	#
	# model to HDF5
	#
	model.save(model_file, include_optimizer=False)

	#
	# weights to HDF5
	#
	model.save_weights(weights_file)

	#
	# architecture to JSON
	#
	#	with open(path+"/architecture.json", 'w') as f:
	#		f.write(model.to_json())


######################
#
# load model (keras)
# https://jovianlin.io/saving-loading-keras-models/
#
def	load_model_by_keras(model_file:str="model.h5", weights_file:str="weights.h5")->keras.models.Model:
	#
	# model to HDF5
	#
	model = keras.models.load_model(model_file, compile=False)

	#
	# weights to HDF5
	#
	model.load_weights(weights_file)

	#
	# architecture from JSON
	#
	#	with open(path+"/architecture.json") as f:
	#		model = keras.models.model_from_json(f.read())

	return model



#####################
#
# upload (gdrive)
# https://github.com/gsuitedevs/PyDrive
# http://www.inmyzakki.com/entry/2017/12/07/190000
# https://qiita.com/i8b4/items/322dc8d81427717a86e4
# https://blog.shikoan.com/google-colab-drive-save/
#
# google colaboratory
#	https://qiita.com/firedfly/items/9b76d4f4ea2b563777af
#
def upload_to_gdrive(name:str)->None:
	#
	# colab
	#
	from google.colab import files

	#
	# Install the PyDrive wrapper & import libraries.
	# This only needs to be done once in a notebook.
	#	!pip install -U -q PyDrive
	#
	from pydrive.auth			import GoogleAuth
	from pydrive.drive			import GoogleDrive
	from google.colab			import auth
	from oauth2client.client	import GoogleCredentials

	#
	# Authenticate and create the PyDrive client.
	# This only needs to be done once in a notebook.
	#
	auth.authenticate_user()
	gauth				= GoogleAuth()
	gauth.credentials	= GoogleCredentials.get_application_default()
	drive				= GoogleDrive(gauth)

	# Create & upload a file.
	uploaded = drive.CreateFile({'title': name})
	uploaded.SetContentFile(name)
	uploaded.Upload()
	print( 'upload_to_gdrive: Uploaded file with ID {}'.format( uploaded.get('id') ) )



#####################
#
# download (gdrive)
#
def download_from_gdrive(fid:str)->None:
	#
	# Install the PyDrive wrapper & import libraries.
	# This only needs to be done once per notebook.
	#	!pip install -U -q PyDrive
	from pydrive.auth			import GoogleAuth
	from pydrive.drive			import GoogleDrive
	from google.colab			import auth
	from oauth2client.client	import GoogleCredentials

	#
	# Authenticate and create the PyDrive client.
	# This only needs to be done once per notebook.
	#
	auth.authenticate_user()
	gauth				= GoogleAuth()
	gauth.credentials	= GoogleCredentials.get_application_default()
	drive				= GoogleDrive(gauth)

	#
	# Download a file based on its file ID.
	#
	# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
	# file_id = '17Lo_ZxYcKO751iYs4XRyIvVXME8Lyc75'
	#
	downloaded = drive.CreateFile({'id': fid})
	print( 'download_from_gdrive: Downloaded content "{}"'.format( downloaded.GetContentString() ) )
	downloaded.GetContentFile('download.bin')	# Download file as 'download.bin'.


#####################
#
# get version
#
def get_version(path:str)->None:
	print('keras version:' + keras.__version__)
	import h5py
	f = h5py.File(path, 'r')
	print('model version:' + f.attrs.get('keras_version').decode('utf-8'))


#####################
#
# env
# os.environ.get('NEW_KEY', 'default')
#


#####################
#
# timeit
#
#	usage:
#		@timeit("sleeper", ndigits=2)
#		def sleeper(sec):
#			time.sleep(sec)
#			return sec
#	out:
#		Excecution time for 'sleeper': 1.24 [sec]
#
"""
import time
from functools import wraps
def timeit(message:str, ndigits:int=2):
	#
	# Print execution time [sec] of function/method
	# - message: message to print with time
	# - ndigits: precision after the decimal point
	#
	def outer_wrapper(func):
		# @wraps: keep docstring of "func"
		@wraps(func)
		def inner_wrapper(*args, **kwargs):
			start = time.time()
			result = func(*args, **kwargs)
			end = time.time()
			print(" 🥑 Excecution time for '{message}': {sec} [sec]".format(
				message=message,
				sec=round(end-start, ndigits))
			)
			return result
		return inner_wrapper
	return outer_wrapper
"""

if __name__ == '__main__':
	pass

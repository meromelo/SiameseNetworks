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

#
#	plaidml
#
import plaidml.keras
plaidml.keras.install_backend()

#
# Standard library imports.
#
import os
import os.path
import subprocess
import sys
import argparse
import glob
import lzma
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

#
# Keras
#
from keras.models				import Sequential, Model
from keras.layers				import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate
from keras.layers				import GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract
from keras.layers.convolutional	import Convolution2D
from keras.layers.pooling		import MaxPooling2D, AveragePooling2D
from keras.activations			import relu, softmax
from keras.optimizers			import Adam, RMSprop, SGD
from keras.regularizers			import l2
from keras						import backend as K
from keras.callbacks			import EarlyStopping
from keras						import models
from keras.utils				import plot_model
import keras.callbacks
import keras.backend.tensorflow_backend as KTF

#
# TensorFlow, sklearn
#
import tensorflow as tf
import sklearn
from sklearn.manifold			import TSNE
from sklearn.decomposition		import PCA

#
# toolz
#
from toolz						import juxt, identity
from operator					import add, mul

#
# funcy
#
#from funcy.py3 import whatever, you, need

#
# Local application/library specific imports.
#
from hyperdash			import Experiment	as hyperdash_experiment
from lib_observe		import hyperdash_callback
from train				import train


#####################
#
#	main
#
def main(argv=sys.argv[1:])->int:
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
	return 0


if __name__ == '__main__':
	sys.exit(main())

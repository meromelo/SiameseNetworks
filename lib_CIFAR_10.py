#!/usr/bin/env python
#!/usr/bin/env PYTHONIOENCODING=utf-8
# encoding: utf-8
# -*- coding: utf-8 -*-
# vim:fileencoding=UTF-8

######################
##
## USER EXPERIENCE DESIGN DEPARTMENT
## DESIGN CENTER
## PRODUCTS BUSINESS STRATEGY DIVISION
## SHARP CORPORATION
##
## SEAVANS SOUTH 18F,
## 1-2-3 SHIBAURA, MINATO-KU, TOKYO, 105-0023, JAPAN
##
## M.Nakazawa ( nakazawa.masayuki@sharp.co.jp )
##
##

#
# Standard library imports.
#
from __future__ import absolute_import, print_function, unicode_literals

import os
import os.path	as path
import shutil
import pickle

#
# Related third party imports.
#
import numpy	as np

from keras.utils			import to_categorical
from keras.utils.data_utils	import get_file
from operator				import attrgetter

#
# Local application/library specific imports.
#


#####################
#
# CLASS
#	CIFAR-10
#	80 million tiny imagesのサブセット
#	Alex Krizhevsky, Vinod Nair, Geoffrey Hintonが収集
#	32x32のカラー画像60000枚
#	10クラスで各クラス6000枚
#	50000枚の訓練画像と10000枚（各クラス1000枚）のテスト画像
#	クラスラベルは排他的
#	PythonのcPickle形式で提供されている
#
#####################
class CIFAR_10:
	#
	# class variables
	#

	######################
	#
	# ctor
	#
	def	__init__(self):
		pass

	######################
	#
	# __download_data
	#
	def	__download_data(self, data_path):
		url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
		shutil.copytree(get_file('cifar-10-batches-py', origin=url, untar=True), data_path)

	######################
	#
	# __load_pickle
	#
	# オリジナルの画像はchannel(3ch) ,row(32pix), column(32pix)のフラット形式3*32*32=3072次元ベクトル
	# reshape() を使って([0]channel, [1]row, [2]column) の3次元arrayに変換
	# numpy のtranspose() を使って imshow() で読める ([1]row, [2]column, [0] channel) の順番に変更
	# (channel, row, column) => (row, column, channel)
	#
	# d = np.array(batch[b'data'], dtype=float)			# shape=(10000, 3072)
	# e = d.reshape(d.shape[0], 3, 32, 32)				# shape=(10000, channel:3, row:32, column:32)
	# f = e.transpose(0, 2, 3, 1)						# shape=(10000, row:32, column:32, channel:3)
	# g = to_categorical(np.array(batch[b'labels']))	# shape=(10000, 10)
	#
	# to_cateogorical関数は正解クラスをone-hotエンコーディングして出力に与えたいときに使うことが多い。
	# from keras.utils.np_utils import to_categorical
	# to_categorical([[1, 3]], num_classes=4)
	# array([[
	# [0., 1., 0., 0.],
	# [0., 0., 0., 1.]]], dtype=float32)
	#
	def	__load_pickle(self, path):
		with open(path, 'rb') as f:
			batch = pickle.load(f, encoding='bytes')
		#
		# datum: (10000,  3, 32, 32) -> (10000, 32, 32, 3) / 255
		# label: (10000, 10)         -> (10000, 10) [one-hot]
		#
		datum = np.array(batch[b'data']).reshape(batch[b'data'].shape[0], 3, 32, 32).transpose(0, 2, 3, 1) / 255
		label = to_categorical(np.array(batch[b'labels']))
		return datum, label

	######################
	#
	# __load_many_pickles
	#
	def	__load_many_pickles(self, paths):
		#
		# zip : zip('ABCD', 'xy') -> Ax By
		#	names = ['Alice', 'Bob', 'Charlie', 'Dave']
		#	ages = [24, 50, 18]
		#	for name, age in zip(names, ages):
		#		print(name, age)
		#	=> Alice 24
		#	=> Bob 50
		#	=> Charlie 18
		#
		#	from itertools import zip_longest
		#	for name, age in zip_longest(names, ages, fillvalue=20):
		#		print(name, age)
		#	=> Alice 24
		#	=> Bob 50
		#	=> Charlie 18
		#	=> Dave 20
		#
		# zip() に続けて * 演算子を使うと、zip したリストを元に戻せます:
		#	=> x = [1, 2, 3]
		#	=> y = [4, 5, 6]
		#	=> zipped = zip(x, y)
		#	=> list(zipped)
		#	=> [(1, 4), (2, 5), (3, 6)]
		#	=> x2, y2 = zip(*zip(x, y))
		#	=> x == list(x2) and y == list(y2)
		#	=> True
		#
		# enumerate & zip
		#	for i,(ai,bi) in enumerate(zip(a,b)): #zipのところを()で囲った
		#		print(i,ai,bi)
		#	=> 0 あ か
		#	=> 1 い き
		#	=> 2 う く
		#	=> 3 え け
		#	=> 4 お こ
		#
		# np.concatenate
		#	import numpy as np
		#	a1 = np.array([[1, 2, 3], [4,  5,  6]])
		#	a2 = np.array([[7, 8, 9], [0, 11, 12]])
		#	print(np.concatenate([a1, a2], axis=0))
		#	=> [[  1  2  3]
		#	=>  [  4  5  6]
		#	=>  [  7  8  9]
		#	=>  [ 10 11 12]]
		#	print(np.concatenate([a1, a2], axis=1))
		#	=> [[ 1  2  3   7  8  9]
		#	=>  [ 4  5  6  10 11 12]]
		#
		# tuple : 「リストは変更ができるが、タプルは変更ができない」、これが両者の違い。
		# タプルの使いどころ 1: 変更を許可しない変数を定義する
		# タプルの使いどころ 2: dict のキーに使う
		# タプルの使いどころ 3: パフォーマンスをよりよくする
		#	tuple1 = ('東京都', '神奈川県', '大阪府')
		#	type(tuple1) == tuple  # => True
		#	tuple1[0]  # => '東京都'
		#	tuple1[1:]  # => ('神奈川県', '大阪府')
		#
		#

		#	paths= [
		#		'./data/data_batch_1',	:(10000, 32, 32, 3)(10000, 10)
		#		'./data/data_batch_2',	:(10000, 32, 32, 3)(10000, 10)
		#		'./data/data_batch_3',	:(10000, 32, 32, 3)(10000, 10)
		#		'./data/data_batch_4',	:(10000, 32, 32, 3)(10000, 10)
		#		'./data/data_batch_5']	:(10000, 32, 32, 3)(10000, 10)
		#
		datum, label = map(np.concatenate, zip(*map(self.__load_pickle, paths)))
		return datum, label


	######################
	#
	# load_data
	#
	# @staticmethod
	#
	# def	load_data(self, data_path='./data'):
	#		if not path.exists(data_path):
	#			self.__download_data(data_path)
	#
	# @staticmethod
	# def	load_data(data_path="./data"):
	# @classmethod
	# def	load_data(cls,data_path="./data"):
	def	load_data(self, data_path="./data"):
		# print(f"CIFAR_10::load_data: data_path={data_path}")
		if not path.exists(data_path):
			print(f"CIFAR_10::load_data: not found data_path={data_path}...download data.")
			self.__download_data(data_path)
		else:
			print(f"CIFAR_10::load_data: found data_path={data_path}")

		# sort
		#	l1 = [(7, 2), (3, 4), (5, 5), (10, 3)]
		#	l2 = sorted(l1, key=lambda x: x[1])
		#	=> [(7, 2), (10, 3), (3, 4), (5, 5)]
		# map(func, iterator)
		#	a = [-1, 3, -5, 7, -9]
		#	print list(map(abs, a))
		#	=> [1, 3, 5, 7, 9]
		# filter
		#	a = [-1, 3, -5, 7, -9]
		#	print list(filter(lambda x: abs(x) > 5, a))
		#	=> [7, -9]

		# f = attrgetter('name') とした後で、f(b) を呼び出すと b.name を返します。
		# f = attrgetter('name', 'date') とした後で、f(b) を呼び出すと (b.name, b.date) を返します。
		# f = attrgetter('name.first', 'name.last') とした後で、f(b) を呼び出すと (b.name.first, b.name.last) を返します。
		#
		# map(func, list)
		#	original_list = list(range(10)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		#	mapped_list = map(lambda x: x**2, original_list)
		#	print(list(mapped_list))
		#	[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

		# file_list = sorted(glob.glob('path/to/dir/*.txt'))

		# os.scandir -> os.DirEntry
		#	name	: scandir() の path 引数に対して相対的な、エントリのベースファイル名です。
		#	path	: os.path.join(scandir_path, entry.name) と等価の、エントリの絶対パス名です。
		#	inode	: 項目の inode 番号を返します。
		#	is_dir(*, follow_symlinks=True)
		#		この項目がディレクトリまたはディレクトリへのシンボリックリンクである場合 True を返します。
		#		項目がそれ以外のファイルやそれ以外のファイルへのシンボリックリンクである場合や、存在しない場合はFalseを返します。
		#		follow_symlinks が False の場合、項目がディレクトリ (シンボリックリンクはたどりません) の場合にのみTrueを
		#		返します。項目がディレクトリ以外のファイルである場合や、項目がもはや存在しない場合は False を返します。
		#	is_file(*, follow_symlinks=True)
		#		この項目がファイルまたはファイルへのシンボリックリンクである場合、 True を返します。
		#		項目がディレクトリやファイル以外の項目へのシンボリックリンクである場合や、存在しない場合は False を返します。
		#		follow_symlinks が False の場合、項目がファイル (シンボリックリンクはたどりません) の場合にのみTrueを返します。
		#		項目がディレクトリやその他のファイル以外の項目である場合や、項目がもはや存在しない場合は False を返します。
		#	is_symlink()
		#		この項目がシンボリックリンクの場合 (たとえ破損していても)、True を返します。項目がディレクトリや
		#		あらゆる種類のファイルの場合、またはもはや存在しない場合は False を返します。
		#	stat(*, follow_symlinks=True)
		#		この項目の stat_result オブジェクトを返します。このメソッドは、デフォルトでシンボリックリンクをたどります。
		#		シンボリックリンクを開始するには、 follow_symlinks=False 引数を追加します。
		#
		# import os
		# from operator import attrgetter
		# f=sorted(map(attrgetter('path'),filter(lambda directory_entry: directory_entry.name.startswith('data_batch_'),os.scandir("./data"))))
		# print(list(f))
		# ['./data/data_batch_1', './data/data_batch_2', './data/data_batch_3', './data/data_batch_4', './data/data_batch_5']
		#

		train_data, train_label = self.__load_many_pickles(
				sorted(
					map(
						attrgetter('path'),
						filter(lambda directory_entry: directory_entry.name.startswith('data_batch_'), os.scandir(data_path))
						)
					)
				)

		validation_data, validation_label = self.__load_pickle('{0}/test_batch'.format(data_path))

		return {'training_data':train_data, 'training_label':train_label, 'validation_data':validation_data, 'validation_label':validation_label}



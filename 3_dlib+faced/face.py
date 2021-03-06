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
# dlib
#	https://pypi.org/project/face_recognition/
#
#
# find face locations by YOLO
#	https://github.com/sthanhng/yoloface
#
#
# find face locations by faced (FaceDetector)
#	https://towardsdatascience.com/faced-cpu-real-time-face-detection-using-deep-learning-1488681c1602
#	https://github.com/iitzco/faced
#	pip install git+https://github.com/iitzco/faced.git
#
#
# ImageAI
#	https://medium.com/@guymodscientist/image-prediction-with-10-lines-of-code-3266f4039c7a
#	https://github.com/OlafenwaMoses/ImageAI
#	pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.3/imageai-2.0.3-py3-none-any.whl 
#

#
# Standard library imports.
#
from __future__ import absolute_import, print_function, unicode_literals

import os
import re
import sys

if os.getenv( 'VIRTUAL_ENV') == None:
	pass

#
# import check
#	try:
#		import d1.d2
#	except ImportError as e:
#		print(e)
#		--- OR ---
#		raise ValueError('stdout argument not allowed, it will be overridden.')
#
#	if system == 'linux':
#		import linuxdeps as deps
#	elif system == 'win32':
#		import win32deps as deps

#
# params
#
global	face_cascade
global	face_detector
global	key_save
global	key_facelandmarks
global	key_mosaic
global	key_quit
global	facelandmarks_ext
key_save				= ' '
key_facelandmarks		= 'm'
key_facelandmarks_ext	= 'e'
key_mosaic				= 'a'
key_quit				= 'q'
facelandmarks_ext		= 0

######################
#
# pillow -> opencv
#	http://tatabox.hatenablog.com/entry/2013/07/21/231751
#	OpenCV_data=np.asarray(PIL_data)
#	https://qiita.com/derodero24/items/f22c22b22451609908ee
#
def pil2cv(image):
	''' PIL型 -> OpenCV型 '''
	# new_image = np.array(image)
	new_image = np.asarray(image)
	if new_image.ndim == 2:  # モノクロ
		pass
	elif new_image.shape[2] == 3:  # カラー
		# new_image = new_image[:, :, ::-1]
		# new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
		pass
	elif new_image.shape[2] == 4:  # 透過
		# new_image = new_image[:, :, [2, 1, 0, 3]]
		# new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
		pass
	return new_image

######################
#
# faceland mark
#	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2
#
def	draw_face_landmarks(frame, top, right, bottom, left, name):
	#
	# scan face landmarks
	# [{
	# 	'chin'			: [(412, 221), (413, 261), (418, 302), (428, 339), (445, 373), (465, 405), (487, 433), (514, 454), (545, 461), (576, 455), (601, 435), (621, 407), (638, 376), (654, 343), (665, 308), (671, 268), (673, 230)],
	# 	'left_eyebrow'	: [(443, 199), (458, 178), (483, 172), (509, 178), (533, 189)],
	# 	'right_eyebrow'	: [(574, 190), (595, 180), (619, 176), (641, 182), (653, 201)],
	# 	'nose_bridge'	: [(554, 226), (555, 253), (556, 279), (557, 307)],
	# 	'nose_tip'		: [(528, 322), (540, 327), (554, 331), (568, 327), (579, 322)],
	# 	'left_eye'		: [(470, 232), (484, 230), (501, 230), (516, 235), (500, 239), (484, 239)],
	# 	'right_eye'		: [(585, 236), (599, 232), (614, 232), (627, 236), (615, 241), (600, 241)],
	# 	'top_lip'		: [(511, 368), (530, 362), (543, 358), (553, 362), (565, 358), (574, 363), (588, 371), (580, 371), (565, 370), (553, 372), (543, 370), (520, 369)],
	# 	'bottom_lip'	: [(588, 371), (574, 383), (564, 388), (552, 390), (541, 388), (529, 382), (511, 368), (520, 369), (542, 371), (553, 373), (564, 372), (580, 371)]
	# }]

	# cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
	# cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

	"""
	font = cv2.FONT_HERSHEY_DUPLEX
	cv2.putText(frame, name, (left+10, top-10), font, 1.0, (0, 255, 0), 1)
	"""

	#
	# PIL image
	#
	# pil_image = PIL.Image.fromarray(frame, 'RGB')
	# d = PIL.ImageDraw.Draw(pil_image, 'RGBA')

	"""
	#
	# box
	#
	x = left
	y = top
	w = right  - left
	h = bottom - top
	s = [(x,y+h/3), (x,y), (x+w/3,y)]
	d.line(s, fill=(0, 255, 0, 255), width=3)
	s = [(x+w-w/3,y), (x+w,y), (x+w,y+h/3)]
	d.line(s, fill=(0, 255, 0, 255), width=3)
	s = [(x,y+h-h/3), (x,y+h), (x+w/3,y+h)]
	d.line(s, fill=(0, 255, 0, 255), width=3)
	s = [(x+w-w/3,y+h), (x+w,y+h), (x+w,y+h-h/3)]
	d.line(s, fill=(0, 255, 0, 255), width=3)
	"""

	#
	# scan face landmarks
	#
	face_landmarks_list = faced.face_landmarks(frame)
	print( f"facelandmarks: list name={name}, len={len(face_landmarks_list)}, face_landmarks={face_landmarks_list}" )

	"""
	clone = frame.copy()
	cv2.putText(clone, "Dlib", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
	# landmarkを画像に書き込む
	for (x, y) in shape[0:68]:
		cv2.circle(clone, (x, y), 1, (0, 0, 255), 5)
		# shapeで指定した個所の切り取り画像(ROI)を取得
		(x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]])) #口の部位のみ切り出し
		roi = frame[y:y + h, x:x + w]
		roi = cv2.resize(roi,(160,100))
		(x, y, w, h) = cv2.boundingRect(np.array([shape[42:48]]))  # 左目の部位のみ切り出し
		leftEye = frame[y:y + h, x:x + w]
		leftEye = cv2.resize(leftEye, (100, 50))
		(x, y, w, h) = cv2.boundingRect(np.array([shape[36:42]]))  # 左目の部位のみ切り出し
		rightEye = frame[y:y + h, x:x + w]
		rightEye = cv2.resize(rightEye, (100, 50))
		return clone, roi
	else :
		return frame, None
	"""

	"""
	# Draw a box around the face
	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

	# Draw a label with a name below the face
	cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
	font = cv2.FONT_HERSHEY_DUPLEX
	cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
	"""

	# cv2.rectangle(save_image, tuple([face_rect.left(),face_rect.top()]), tuple([face_rect.right(),face_rect.bottom()]), (0, 0,255), thickness=2)
	# "指定したイメージの領域を塗りつぶす"
	# points = cv2.convexHull(points)
	# cv2.fillConvexPoly(image, points, color = color)

	pil_image = PIL.Image.fromarray(frame, 'RGB')
	d = PIL.ImageDraw.Draw(pil_image, 'RGBA')

	count = 0
	for face_landmarks in face_landmarks_list:
		print( f"facelandmarks: [{count}] name={name}, len={len(face_landmarks)}, face_landmarks={face_landmarks}" )
		#
		# draw face landmarks
		#
		global facelandmarks_ext
		if facelandmarks_ext == 0:
			# 	'chin'			: [(412, 221), (413, 261), (418, 302), (428, 339), (445, 373), (465, 405), (487, 433), (514, 454), (545, 461), (576, 455), (601, 435), (621, 407), (638, 376), (654, 343), (665, 308), (671, 268), (673, 230)],
			# 	'left_eyebrow'	: [(443, 199), (458, 178), (483, 172), (509, 178), (533, 189)],
			# 	'right_eyebrow'	: [(574, 190), (595, 180), (619, 176), (641, 182), (653, 201)],
			# 	'nose_bridge'	: [(554, 226), (555, 253), (556, 279), (557, 307)],
			# 	'nose_tip'		: [(528, 322), (540, 327), (554, 331), (568, 327), (579, 322)],
			# 	'left_eye'		: [(470, 232), (484, 230), (501, 230), (516, 235), (500, 239), (484, 239)],
			# 	'right_eye'		: [(585, 236), (599, 232), (614, 232), (627, 236), (615, 241), (600, 241)],
			# 	'top_lip'		: [(511, 368), (530, 362), (543, 358), (553, 362), (565, 358), (574, 363), (588, 371), (580, 371), (565, 370), (553, 372), (543, 370), (520, 369)],
			# 	'bottom_lip'	: [(588, 371), (574, 383), (564, 388), (552, 390), (541, 388), (529, 382), (511, 368), (520, 369), (542, 371), (553, 373), (564, 372), (580, 371)]
			offset = 4
			for c in face_landmarks['chin']:
				d.arc(c+(c[0]+offset,c[1]+offset), start=0, end=360, fill=(255, 255, 0))
			for c in face_landmarks['left_eyebrow']:
				d.arc(c+(c[0]+offset,c[1]+offset), start=0, end=360, fill=(255, 255, 0))
			for c in face_landmarks['right_eyebrow']:
				d.arc(c+(c[0]+offset,c[1]+offset), start=0, end=360, fill=(255, 255, 0))
			for c in face_landmarks['nose_bridge']:
				d.arc(c+(c[0]+offset,c[1]+offset), start=0, end=360, fill=(255, 255, 0))
			for c in face_landmarks['nose_tip']:
				d.arc(c+(c[0]+offset,c[1]+offset), start=0, end=360, fill=(255, 255, 0))
			for c in face_landmarks['left_eye']:
				d.arc(c+(c[0]+offset,c[1]+offset), start=0, end=360, fill=(255, 255, 0))
			for c in face_landmarks['right_eye']:
				d.arc(c+(c[0]+offset,c[1]+offset), start=0, end=360, fill=(255, 255, 0))
			for c in face_landmarks['top_lip']:
				d.arc(c+(c[0]+offset,c[1]+offset), start=0, end=360, fill=(255, 255, 0))
			for c in face_landmarks['bottom_lip']:
				d.arc(c+(c[0]+offset,c[1]+offset), start=0, end=360, fill=(255, 255, 0))

		elif facelandmarks_ext == 1:
			d.polygon(face_landmarks['left_eye' ], fill=None, outline=(255,255,255))
			d.polygon(face_landmarks['right_eye'], fill=None, outline=(255,255,255))

			eL = [face_landmarks['left_eye'][0]] + face_landmarks['left_eyebrow'] + [face_landmarks['left_eye'][3]] + [face_landmarks['left_eye'][4]] + [face_landmarks['left_eye'][5]]
			d.polygon(eL, fill=None, outline=(255,255,255))
			eR = [face_landmarks['right_eye'][0]] + face_landmarks['right_eyebrow'] + [face_landmarks['right_eye'][3]] + [face_landmarks['right_eye'][4]] + [face_landmarks['right_eye'][5]]
			d.polygon(eR, fill=None, outline=(255,255,255))

			aL = [face_landmarks['left_eye' ][3]] + [face_landmarks['nose_bridge'][-1]] + [face_landmarks['nose_tip'][ 0]]
			d.polygon(aL, fill=None, outline=(255,255,255))
			aR = [face_landmarks['right_eye'][0]] + [face_landmarks['nose_bridge'][-1]] + [face_landmarks['nose_tip'][-1]]
			d.polygon(aR, fill=None, outline=(255,255,255))

			bL = [face_landmarks['left_eye' ][4]] + [face_landmarks['top_lip'][0]] + [face_landmarks['nose_tip'][ 0]] + [face_landmarks['left_eye' ][3]]
			d.polygon(bL, fill=None, outline=(255,255,255))
			bR = [face_landmarks['right_eye'][5]] + [face_landmarks['top_lip'][6]] + [face_landmarks['nose_tip'][-1]] + [face_landmarks['right_eye'][0]]
			d.polygon(bR, fill=None, outline=(255,255,255))

			cL = [face_landmarks['top_lip'][0]] + [face_landmarks['nose_bridge'][-1]] + [face_landmarks['nose_tip'][ 0]]
			d.polygon(cL, fill=None, outline=(255,255,255))
			cR = [face_landmarks['top_lip'][6]] + [face_landmarks['nose_bridge'][-1]] + [face_landmarks['nose_tip'][-1]]
			d.polygon(cR, fill=None, outline=(255,255,255))

			d.polygon(face_landmarks['top_lip'   ], fill=None, outline=(255,255,255))
			d.polygon(face_landmarks['bottom_lip'], fill=None, outline=(255,255,255))

		elif facelandmarks_ext == 2:
			# Make the eyebrows into a nightmare
			d.line(face_landmarks['left_eyebrow'], fill=(0, 0, 0, 255), width=3)
			d.line(face_landmarks['right_eyebrow'], fill=(0, 0, 0, 255), width=3)
			d.polygon(face_landmarks['left_eyebrow'], fill=(0, 0, 0, 255))
			d.polygon(face_landmarks['right_eyebrow'], fill=(0, 0, 0, 255))

			# Gloss the lips
			d.line(face_landmarks['top_lip'], fill=(0, 0, 0, 255), width=10)
			d.line(face_landmarks['bottom_lip'], fill=(0, 0, 0, 255), width=10)

			d.polygon(face_landmarks['bottom_lip'], fill=(255, 0, 0, 255))
			d.polygon(face_landmarks['top_lip'], fill=(255, 0, 0, 255))
			d.line(face_landmarks['top_lip'], fill=(0, 0, 0, 255), width=2)
			d.line(face_landmarks['bottom_lip'], fill=(0, 0, 0, 255), width=2)

			# Chin
			d.polygon(face_landmarks['chin'], fill=(255, 0, 0, 16))

			# Apply some eyeliner
			d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(10, 0, 0, 255), width=6)
			d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(10, 0, 0, 255), width=6)

			# Sparkle the eyes
			d.polygon(face_landmarks['left_eye'], fill=(255, 0, 0, 200))
			d.polygon(face_landmarks['right_eye'], fill=(255, 0, 0, 200))

		count +=1
		break

	return pil2cv(pil_image)

######################
#
# scan_known_people
#
def scan_known_people(folder):
	#
	# http location
	#
	# import requests
	# from io import BytesIO
	# response = requests.get("https://raw.githubusercontent.com/solegaonkar/solegaonkar.github.io/master/img/rahul1.jpeg")
	# frame = faced.load_image_file(BytesIO(response.content))
	# face_locations = faced.face_locations(frame)
	#
	if not os.path.exists(folder):
		# print(f'scan_known_people: no such folder={folder}')
		# raise ValueError('no such foler.')
		return [], []

	print(f'scan_known_people: ', end="")
	known_names				= []
	known_face_encodings	= []
	for file in image_files_in_folder(folder):
		basename	= os.path.splitext(os.path.basename(file))[0]
		img			= faced.load_image_file(file)
		encodings	= faced.face_encodings(img)

		if len(encodings) > 1:
			print(f"WARNING: More than one face found in {file}. Only considering the first face.")

		if len(encodings) == 0:
			print(f"WARNING: No faces found in {file}. Ignoring file.")
		else:
			known_names.append(basename)
			known_face_encodings.append(encodings[0])
			print(f'{basename}, ', end="")

	print(f'.')
	return known_names, known_face_encodings

######################
#
# image_files_in_folder
#
def image_files_in_folder(folder):
	return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

######################
#
# find face locations by haarcascade
#
def face_locations_by_haarcascade(frame, cascade):
	gray	= cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	faces	= cascade.detectMultiScale(gray, 1.3, 5)

	#
	#	common_faces
	#
	common_faces	= []
	padding			= 5;
	for (x, y, w, h) in faces:
		x1 = x     - padding
		y1 = y     - padding
		x2 = x + w + padding
		y2 = y + h + padding
		common_faces.append( (y1, x2, y2, x1) )	# # top, right, bottom, left
	return common_faces

######################
#
# find face locations by YOLO
#	https://github.com/sthanhng/yoloface
#
def face_locations_by_YOLO(frame, YOLOnet):

	# Create a 4D blob from a frame.
	# blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
	blob = cv2.dnn.blobFromImage(frame, 1 / 255, (YOLOu.IMG_WIDTH, YOLOu.IMG_HEIGHT), [0, 0, 0], 1, crop=False)

	# Sets the input to the network
	YOLOnet.setInput(blob)

	# Runs the forward pass to get output of the output layers
	# outs = YOLOnet.forward(get_outputs_names(YOLOnet))
	outs = YOLOnet.forward(YOLOu.get_outputs_names(YOLOnet))

	# Remove the bounding boxes with low confidence
	# faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
	faces = YOLOu.post_process(frame, outs, YOLOu.CONF_THRESHOLD, YOLOu.NMS_THRESHOLD)

	#
	#	common_faces
	#
	common_faces	= []
	padding			= 5;
	for (x, y, w, h) in faces:
		x1 = x     - padding
		y1 = y     - padding
		x2 = x + w + padding
		y2 = y + h + padding
		# cv2.imshow('Video', img)
		common_faces.append( (y1, x2, y2, x1) )	 # top, right, bottom, left
		# print (f"haar faces:{faces}")
	return common_faces

######################
#
# find face locations by faced (FaceDetector)
#	https://towardsdatascience.com/faced-cpu-real-time-face-detection-using-deep-learning-1488681c1602
#	https://github.com/iitzco/faced
#	pip install git+https://github.com/iitzco/faced.git
#
def face_locations_by_FaceDetector(frame):
	#
	# find faces
	#	return: x, y, w, h, ?
	global	face_detector
	faces	= face_detector.predict(frame, thresh=0.85)
	# print (f"FaceDetector faces:{faces}")

	#
	#	common_faces
	#
	common_faces	= []
	padding			= 5;
	for (x, y, w, h, _) in faces:
		x1 = int(x-w/2) - padding
		y1 = int(y-h/2) - padding
		x2 = int(x+w/2) + padding
		y2 = int(y+h/2) + padding
		common_faces.append( (y1,x2,y2,x1) )	# top, right, bottom, left 
		# img = cv2.rectangle( frame, (x1,y1), (x2,y2), (0,255,0), 2 )

	# print (f"FaceDetector common_faces:{common_faces}")
	return common_faces

######################
#
# mosaic
#
def mosaic(src, ratio=0.1):
	small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
	#
	# debug
	#	cv2.imshow("video", small)
	#	s = cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
	#	cv2.imshow("video", s)
	#	return s
	return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(src, x, y, w, h, ratio=0.1):
	dst = src.copy()
	dst[y:y+h, x:x+w] = mosaic(dst[y:y+h, x:x+w], ratio)
	return dst

######################
#
# main
#	parser.add_argument('-m', '--mode',		type=int,	default=0, choice=[0,1,2,3], help='detection algorithm. [0:hog, 1:haar, 2:YOLO, 3:faced]')
#	parser.add_argument('-c', '--cfg',		type=str,	default='YOLOv3-face.cfg', help='if mode==YOLO, path to config file')
#	parser.add_argument('-f', '--weights',	type=str,	default='YOLOv3-wider_16000.weights', help='if mode==YOLO, path to weights of model')
#
def	main(mode:int=0, device:int=0, size:int=480, cfg:str='YOLOv3-cfg', weights:str='YOLOv3-wider_16000.weights', ratio:float=1.0, facemarks:bool=False, mosaic:bool=False)->None:
	#
	# detection algorithm [0:hog, 1:haar, 2:YOLO, 3:faced]
	#
	import cv2
	if mode == 0:
		print( f'detecting faces: dlib hog/cnn.' )

	elif mode == 1:
		print( f'detecting faces: AdaBoost using haarcascade.' )
		global	face_cascade
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	elif mode == 2:
		print( f'detecting faces: YOLOv3.' )
		global	YOLOnet
		YOLOnet = cv2.dnn.readNetFromDarknet(cfg, weights)
		YOLOnet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
		YOLOnet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

	elif mode == 3:
		print( f'detecting faces: faced(FaceDetector).' )
		global face_detector
		face_detector = FaceDetector()

	else:
		print( f'detecting faces: unknown={mode}.' )
		return

	#
	# window for video capture
	#
	# import cv2
	windowName	= 'Video'
	cv2.namedWindow(windowName)
	# cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
	cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.waitKey(10)
	cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


	#
	# vide capture
	# Macbook12: fps=30.0, w=848.0, h=480.0
	# iMac5K   : fps=29.97002997002997, w=960.0, h=544.0, detecting faces: YOLOv3.
	#
	print( f'video capture device: {device}' )
	video_capture = cv2.VideoCapture(device)
	if not video_capture.isOpened():
		print(f'Couldnt open video file or webcam, device={device}.')
		raise ImportError(f'Couldnt open video file or webcam, device={device}.')

	video_capture.set(cv2.CAP_PROP_FPS, 60)           		# カメラFPSを60FPSに設定
	video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, size)		# カメラ画像の横幅を1280に設定 (640)
	# video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size)	# カメラ画像の縦幅を720に設定 (480)
	print(f"video capture device: fps={video_capture.get(cv2.CAP_PROP_FPS)}")
	print(f"video capture device: w={video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)}")
	print(f"video capture device: h={video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

	#
	# count
	#
	count = 0

	#
	# down sampling ratio
	#
	down_sampling_ratio	= 1.0 / ratio

	#
	# fps
	#
	tm	= cv2.TickMeter()
	tm.start()
	fps_count		= 0
	fps_count_max	= 10
	fps_number		= 0

	#
	# capture video frame
	#
	while(video_capture.isOpened() == True):
		#
		# grab a single frame of video
		#
		ret, frame = video_capture.read()
		if not ret:
			print( f"-" )
			continue
		# print( f"." )

		#
		# flip
		#
		frame = cv2.flip(frame, 1) # Flip camera horizontaly

		#
		# Resize frame of video to 1/4 size for faster face recognition processing
		# frame_buffer = imutils.resize(frame, width=480)
		#
		if ratio > 1.0:
			# frame_buffer = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
			frame_buffer = cv2.resize(frame, (0, 0), fx=down_sampling_ratio, fy=down_sampling_ratio)
		else:
			frame_buffer = frame

		#
		# convert the image from BGR color (which OpenCV uses) to RGB color (which faced uses)
		# rgb_frame = frame[:, :, ::-1]
		# rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		#
		rgb_frame = cv2.cvtColor(frame_buffer, cv2.COLOR_BGR2RGB)

		#
		# Find all the faces and face enqcodings in the frame of video
		#
		face_locations	= []
		if mode == 0:
			name			= "hog"	# hog or cnn
			face_locations	= faced.face_locations(rgb_frame, number_of_times_to_upsample=1, model=name)
		elif mode == 1:
			# global	face_cascade
			face_locations	= face_locations_by_haarcascade(rgb_frame, face_cascade)

		elif mode == 2:
			# global	YOLOnet
			face_locations	= face_locations_by_YOLO(rgb_frame, YOLOnet)

		elif mode == 3:
			face_locations	= face_locations_by_FaceDetector(rgb_frame)

		else:
			print( f'detecting faces: unknown={mode}.' )
			return

		#
		# check faces: (top, right, bottom, left)
		#
		if len(face_locations) > 0:
			x = face_locations[0][3]		# x		: left
			y = face_locations[0][0]		# y		: top
			w = face_locations[0][1] - x	# x + w	: right
			h = face_locations[0][2] - y	# y + h : bottom
			# print( f"face_locations={face_locations}, 0:x={x:3d}, y={y:3d}, w={w:3d}, h={h:3d}", end="")
			print( f'face_locations:{len(face_locations):2d} persons, x={x:3d}, y={y:3d}, w={w:3d}, h={h:3d} : ', end='\n')

		#
		# face encoding features(vector)
		#
		face_encodings	= faced.face_encodings(rgb_frame, face_locations)

		#
		# Loop through each face in this frame of video
		#
		for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
			#
			# See if the face is a match for the known face(s)
			#
			matches = faced.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

			#
			# If a match was found in known_face_encodings, just use the first one.
			# if True in matches:
			#     first_match_index = matches.index(True)
			#     name = known_face_names[first_match_index]
			# Or instead, use the known face with the smallest distance to the new face
			#
			name				= "Unknown"
			face_distances		= faced.face_distance(known_face_encodings, face_encoding)
			best_match_index	= np.argmin(face_distances)
			if matches[best_match_index]:
				name = known_face_names[best_match_index]
				# print( f" !!!found...{name} d({best_match_index})={face_distances[best_match_index]}", end="\n" )
				# print( f"{name}\td[{best_match_index:2d}]={face_distances[best_match_index]:6.4f},\t", end="" )
				print( f"{name}:d[{best_match_index:2d}]={face_distances[best_match_index]:6.4f},\t", end="\n" )
				#
				# face landmarks
				#
				if facemarks == True:
					rgb_frame = draw_face_landmarks(rgb_frame, top, right, bottom, left, name)
					pass
			else:
				# print( f"1 {name}\td[{best_match_index:2d}]={face_distances[best_match_index]:6.4f},\t", end="\n" )
				pass

			# if facemarks == True:
			# 	rgb_frame = draw_face_landmarks(rgb_frame, top, right, bottom, left, name)

			#
			# Scale back up face locations since the frame we detected in was scaled to 1/4 size
			#
			if ratio > 1.0:
				top		*= int(ratio)
				right	*= int(ratio)
				bottom	*= int(ratio)
				left	*= int(ratio)

			#
			# mosaic
			#
			if mosaic == True:
				rgb_frame = mosaic_area(rgb_frame, left, top, right-left, bottom-top, ratio=0.05)

			#
			# Draw a box,label(name) around the face
			#
			print(f'facemarks={facemarks}, name={name}')
			if facemarks == False:
				cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 0, 255), 2)
				cv2.rectangle(rgb_frame, (left, bottom-35), (right, bottom), (0, 0, 255), cv2.FILLED)
				font = cv2.FONT_HERSHEY_DUPLEX
				cv2.putText(rgb_frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
			else:
				x = left
				y = top
				w = right  - left
				h = bottom - top
				# img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
				s = [(x,y+int(h/3)), (x,y), (x+int(w/3),y)]
				cv2.line(rgb_frame, s[0], s[1], color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
				cv2.line(rgb_frame, s[1], s[2], color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

				s = [(x+w-int(w/3),y), (x+w,y), (x+w,y+int(h/3))]
				cv2.line(rgb_frame, s[0], s[1], color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
				cv2.line(rgb_frame, s[1], s[2], color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
				
				s = [(x,y+h-int(h/3)), (x,y+h), (x+int(w/3),y+h)]
				cv2.line(rgb_frame, s[0], s[1], color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
				cv2.line(rgb_frame, s[1], s[2], color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
				
				s = [(x+w-int(w/3),y+h), (x+w,y+h), (x+w,y+h-int(h/3))]
				cv2.line(rgb_frame, s[0], s[1], color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
				cv2.line(rgb_frame, s[1], s[2], color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

				font = cv2.FONT_HERSHEY_DUPLEX
				cv2.putText(rgb_frame, name, (left+6, bottom-6), font, 1.0, (0, 255, 0), 1, cv2.LINE_AA)

		if len(face_locations) > 0:
			print(f'.')

		#
		# fps
		#
		if fps_count == fps_count_max:
			tm.stop()
			fps_number	= fps_count_max / tm.getTimeSec()
			tm.reset()
			tm.start()
			fps_count	= 0

		cv2.putText(rgb_frame, 'FPS:{:.2f}'.format(fps_number), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
		fps_count += 1

		# Display the resulting image
		if ret:
			frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
			cv2.imshow(windowName, frame)
			pass

		#
		# keyboard operation
		#
		global key_save
		global key_facelandmarks
		global key_mosaic
		global key_quit

		c = cv2.waitKey(1) & 0xff

		if c == ord(key_quit) or c == 27:
			break

		elif c == ord(key_save):	# spaceで保存 32
			# cv2.imwrite( './filename_' + str(count) + '.jpg', frame ) #001~連番で保存
			bewrite		= True
			filename	= "screenshot_" + str(count) + ".jpg"
			while os.path.exists(filename):
				print('skip:' + filename)
				count += 1
				filename = "screenshot_" + str(count) + ".jpg"
				if count >= 100:
					beWrite = False

			if bewrite:
				cv2.imwrite( filename, frame ) #001~連番で保存
				count += 1
				print('save done:' + filename)

		elif c == ord(key_facelandmarks):
			facemarks = not facemarks

		elif c == ord(key_mosaic):
			mosaic = not mosaic

		elif c == ord(key_facelandmarks_ext):
			global facelandmarks_ext
			facelandmarks_ext += 1
			_, facelandmarks_ext = divmod(facelandmarks_ext, 3)

	#
	# release handle to the webcam
	#
	video_capture.release()
	cv2.destroyWindow(windowName)
	cv2.destroyAllWindows()


######################
#
# main
#
if __name__ == '__main__':
	#
	# sys.argv
	#
	import sys
	print(f'sys.argv: {sys.argv}')
	print(f'sys.argc: {len(sys.argv)}')
	for (i, arg) in enumerate(sys.argv):
		print(f'debug - sys.argv[{i}]={sys.argv[i]}')

	#
	# args
	#
	#	parser.add_argument('--flag', action='store_true')
	#	parser.add_argument('--fruit', choices=['apple', 'banana', 'orange'])
	#	parser.add_argument('--colors', nargs='*')						# python test.py --colors red green blue
	#	parser.add_agument("-a", required=True)							# required
	#	parser.add_argument('--number', type=int)    					# 整数値(int)
	#	parser.add_argument('--alpha', type=float, default=0.01)    	# 実数値(float)
	#	p = lambda x:list(map(int, x.split('.')))
	#	parser.add_argument('--address', type=tp, help='IP address')	# python test.py --address 192.168.31.150
	#	parser.add_argument('-r', '--resize',	action="store_true", help="resize video capture frame for speed")
	#
	# parser.add_argument('-c', '--cascade',				default=False, action='store_true', help='detecting faces by adaboost haarcascade')
	# parser.add_argument('-y', '--YOLOv3',					default=False, action='store_true', help='detecting faces by YOLOv3')
	# parser.add_argument('-e', '--facedetector',			default=False, action='store_true', help='detecting faces by FaceDetector')
	# p.add_argument('-a',  type=int, choices=[1,2,3])   #整数1,2,3のどれか
	# p.add_argument("-n", const=9, nargs="?")  #nargsは引数の数が0 or 1
	#
	# 設定ファイルの読み込みを切り替える例
	#	def get_config(key):
	#		env = os.environ.get("PYTHON_ENV")
	#		with open("./config/%s.json" % env) as f:
	#		return json.load(f)[key]
	#
	import argparse
	parser = argparse.ArgumentParser()
	"""
	parser.add_argument('-m', '--mode',		type=int,	default=0, choices=[0,1,2,3], help='detection algorithm. [0:hog, 1:haar, 2:YOLO, 3:faced]')
	parser.add_argument('-d', '--device',	type=int,	default=0, help='device id of source camera')
	parser.add_argument('-s', '--size',		type=int,	default=480, help='width size of the camera')
	parser.add_argument('-p', '--pictures',	type=str,	default='pictures/jpg', help='path for known people folder')
	parser.add_argument('-c', '--cfg',		type=str,	default='YOLOv3-face.cfg', help='if mode==YOLO, path to config file')
	parser.add_argument('-f', '--weights',	type=str,	default='YOLOv3-wider_16000.weights', help='if mode==YOLO, path to weights of model')
	parser.add_argument('-r', '--ratio',	type=float,	default=1.0, help="resize video capture frame for speed")
	"""
	"""
	parser.add_argument('-m', type=int,	default=0, choices=[0,1,2,3], help='detection algorithm. [0:hog, 1:haar, 2:YOLO, 3:faced]')
	parser.add_argument('-d', type=int,	default=0, help='device id of source camera')
	parser.add_argument('-s', type=int,	default=480, help='width size of the camera')
	parser.add_argument('-p', type=str,	default='pictures/jpg', help='path for known people folder')
	parser.add_argument('-c', type=str,	default='YOLOv3-face.cfg', help='if mode==YOLO, path to config file')
	parser.add_argument('-f', type=str,	default='YOLOv3-wider_16000.weights', help='if mode==YOLO, path to weights of model')
	parser.add_argument('-r', type=float,	default=1.0, help="resize video capture frame for speed")
	"""
	parser.add_argument('--mode',		type=int,	default=3, choices=[0,1,2,3], help='detection algorithm. [0:hog, 1:haar, 2:YOLO, 3:faced]')
	parser.add_argument('--device',		type=int,	default=0, help='device id of source camera')
	parser.add_argument('--size',		type=int,	default=480, help='width size of the camera')
	parser.add_argument('--pictures',	type=str,	default='pictures/jpg', help='path for known people folder')
	parser.add_argument('--cfg',		type=str,	default='YOLOv3-face.cfg', help='if mode==YOLO, path to config file')
	parser.add_argument('--weights',	type=str,	default='YOLOv3-wider_16000.weights', help='if mode==YOLO, path to weights of model')
	parser.add_argument('--ratio',		type=float,	default=1.0, help="resize video capture frame for speed")
	parser.add_argument('--facemarks',				default=False, action='store_true', help='display face landmarks')
	parser.add_argument('--mosaic',					default=False, action='store_true', help='display mosic face')
	args = parser.parse_args()

	#
	# args
	#
	print(f'python {sys.argv[0]}')
	print(f'  key: quit="{key_quit}", save="{key_save}", facelandmarks="{key_facelandmarks}", mosaic="{key_mosaic}"')
	print(f'  face detection mode:{args.mode}')

	#
	# import
	#	detection algorithm [0:hog, 1:haar, 2:YOLO, 3:faced]
	#
	import PIL.Image
	import PIL.ImageDraw
	import imutils
	import cv2
	import numpy as np
	import dlib
	if (args.mode == 0) or (args.mode == 1):
		pass

	elif args.mode == 2:
		import YOLOu as YOLOu

	elif args.mode == 3:
		from faced import FaceDetector
		# from faced.utils import annotate_image

	else:
		pass

	if os.getenv( 'VIRTUAL_ENV') == None:
		pass

	#
	# dlib
	#	create arrays of known face encodings and their names
	#
	import face_recognition as faced
	print(f'known people foler: "{args.pictures}"')
	known_face_names, known_face_encodings	= scan_known_people(args.pictures)

	if len(known_face_names) == 0:
		print(f'cannot scan known people: "{args.pictures}"')
		sys.exit(1)

	#
	# main function
	#
	main( args.mode, args.device, args.size, args.cfg, args.weights, args.ratio, args.facemarks, args.mosaic )

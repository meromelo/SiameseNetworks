
import face_recognition
import cv2
import numpy as np
import os
import re
import sys

import PIL.Image
import PIL.ImageDraw
import imutils

# import requests
# from io import BytesIO

# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


#
# mosaic
#
def mosaic(src, ratio=0.1):
	small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
	return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(src, x, y, width, height, ratio=0.1):
	dst = src.copy()
	dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
	return dst
#
# image_files_in_folder
#
def image_files_in_folder(folder):
	return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


#
# print_result
#
def print_result(filename, name, distance, show_distance=False):
	if show_distance:
		print(f"{filename},{name},{distance}")
	else:
		print(f"{filename},{name}")

#
# faceland mark
#	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2
#
def	draw_face_landmarks(fr_image, top, right, bottom, left):
	#
	# scan face landmarks
	#
	face_landmarks_list = face_recognition.face_landmarks(fr_image)
	print( f"face_landmarks={len(face_landmarks_list)}" )

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


	# pil_image = PIL.Image.fromarray(fr_image)
	for face_landmarks in face_landmarks_list:
		cv2.rectangle(fr_image, face_landmarks['left_eyebrow'], color=(0,0,0,255), thickness=10)

		"""
		d = PIL.ImageDraw.Draw(pil_image, 'RGBA')
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

		# from IPython.display import display
		# display(pil_image)
		pil_image.show()
		"""

#
# scan_known_people
#
def scan_known_people(known_people_folder):
	known_names				= []
	known_face_encodings	= []

	#
	# http location
	#
	# import requests
	# from io import BytesIO
	# response = requests.get("https://raw.githubusercontent.com/solegaonkar/solegaonkar.github.io/master/img/rahul1.jpeg")
	# fr_image = face_recognition.load_image_file(BytesIO(response.content))
	# face_locations = face_recognition.face_locations(fr_image)
	#

	for file in image_files_in_folder(known_people_folder):
		basename	= os.path.splitext(os.path.basename(file))[0]
		img			= face_recognition.load_image_file(file)
		encodings	= face_recognition.face_encodings(img)

		if len(encodings) > 1:
			print(f"WARNING: More than one face found in {file}. Only considering the first face.")

		if len(encodings) == 0:
			print(f"WARNING: No faces found in {file}. Ignoring file.")
		else:
			known_names.append(basename)
			known_face_encodings.append(encodings[0])

	return known_names, known_face_encodings

#
# test_image
# [test_image(image_file, known_face_names, known_face_encodings, tolerance=0.6, show_distance=True) for image_file in image_files_in_folder(image_to_check)]
#
def test_image(image_to_check, known_names, known_face_encodings, tolerance=0.6, show_distance=False):
	unknown_image = face_recognition.load_image_file(image_to_check)

	# Scale down image if it's giant so things run a little faster
	if max(unknown_image.shape) > 1600:
		pil_img = PIL.Image.fromarray(unknown_image)
		pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
		unknown_image = np.array(pil_img)

	unknown_encodings = face_recognition.face_encodings(unknown_image)

	for unknown_encoding in unknown_encodings:
		distances	= face_recognition.face_distance(known_face_encodings, unknown_encoding)
		result		= list(distances <= tolerance)

		if True in result:
			[print_result(image_to_check, name, distance, show_distance) for is_match, name, distance in zip(result, known_names, distances) if is_match]
		else:
			print_result(image_to_check, "unknown_person", None, show_distance)

	if not unknown_encodings:
		# print out fact that no faces were found in image
		print_result(image_to_check, "no_persons_found", None, show_distance)


######################
#
# main
#
# Create arrays of known face encodings and their names
known_people_folder = "pictures/jpg"
known_face_names, known_face_encodings = scan_known_people(known_people_folder)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
#if not video_capture.isOpened():
#	print( f"cannot open camera, exited.")
#	sys.exit()
if not video_capture.isOpened():
	raise ImportError("Couldn't open video file or webcam.")

video_capture.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,  1280) # カメラ画像の横幅を1280に設定
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,  720) # カメラ画像の縦幅を720に設定
print(f"fps={video_capture.get(cv2.CAP_PROP_FPS)}")
print(f"w={video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"h={video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

#
# count
#
count = 0

#
# isFast
#
isFast		= False
downRatio	= 2.0
ratioValue	= 1.0 / downRatio
print( f"isFast={isFast}, downRatio={downRatio}, ratioValue={ratioValue}" )



# while True:
while(video_capture.isOpened() == True):
	#
	# Grab a single frame of video
	#
	ret, frame = video_capture.read()
	if not ret:
		print( f"-" )
		continue
	print( f"." )

	#
	# Resize frame of video to 1/4 size for faster face recognition processing
	# frame_buffer = imutils.resize(frame, width=480)
	#
	if isFast == True:
		# frame_buffer = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
		frame_buffer = cv2.resize(frame, (0, 0), fx=ratioValue, fy=ratioValue)
	else:
		frame_buffer = frame


	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	# rgb_frame = frame[:, :, ::-1]
	# rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb_frame = frame_buffer[:, :, ::-1]

	# Find all the faces and face enqcodings in the frame of video
	#	face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=1, model='cnn')
	#	face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=1, model='hog')
	face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=1, model='hog')
	face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

	# Loop through each face in this frame of video
	for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
		# See if the face is a match for the known face(s)
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

		name = "Unknown"

		# If a match was found in known_face_encodings, just use the first one.
		# if True in matches:
		#     first_match_index = matches.index(True)
		#     name = known_face_names[first_match_index]
		# Or instead, use the known face with the smallest distance to the new face
		#
		face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
		best_match_index = np.argmin(face_distances)
		if matches[best_match_index]:
			name = known_face_names[best_match_index]
			print( f"found...{name} d({best_match_index})={face_distances[best_match_index]}" )
			# face landmarks
			# draw_face_landmarks(frame, top, right, bottom, left)

		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
		if isFast == True:
			top		*= int(downRatio)
			right	*= int(downRatio)
			bottom	*= int(downRatio)
			left	*= int(downRatio)

		# Draw a box around the face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# Draw a label with a name below the face
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	# Display the resulting image
	if ret:
		cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	c = cv2.waitKey(1) & 0xFF
	if c == ord('q'):
		break
	if c == 32:	#spaceで保存
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
	if c == 27:
		break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


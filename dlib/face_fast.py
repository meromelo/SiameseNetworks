
import face_recognition
import cv2
import numpy as np
import os
import re

import PIL.Image
import PIL.ImageDraw

# import requests
# from io import BytesIO

# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

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

#
# main
#
# Create arrays of known face encodings and their names
known_people_folder = "pictures/jpg"
known_face_names, known_face_encodings = scan_known_people(known_people_folder)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

isFast		= True
downRatio	= 2.0
ratioValue	= 1.0 / downRatio
print( f"isFast={isFast}, downRatio={downRatio}, ratioValue={ratioValue}" )

while True:
	# Grab a single frame of video
	ret, frame = video_capture.read()
	print( f"." )

	# Resize frame of video to 1/4 size for faster face recognition processing
	if isFast == True:
		# frame_buffer = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
		frame_buffer = cv2.resize(frame, (0, 0), fx=ratioValue, fy=ratioValue)
	else:
		frame_buffer = frame

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	# rgb_frame = frame[:, :, ::-1]
	rgb_frame = frame_buffer[:, :, ::-1]

	# Find all the faces and face enqcodings in the frame of video
	face_locations = face_recognition.face_locations(rgb_frame)
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
	cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

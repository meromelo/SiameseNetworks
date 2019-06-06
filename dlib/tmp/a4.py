import face_recognition
picture_of_obama = face_recognition.load_image_file("C:/pytools/face_recognition/known_people/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(picture_of_obama)[0]

another_picture = face_recognition.load_image_file("C:/pytools/face_recognition/unknown_pictures/104.png")
another_face_encoding = face_recognition.face_encodings(another_picture)[0]

results = face_recognition.compare_faces([obama_face_encoding], another_face_encoding)
print(results)

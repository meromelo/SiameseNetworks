import face_recognition
import PIL
import cv2

F = "C:/pytools/face_recognition/unknown_pictures/two_people.jpg"

image = face_recognition.load_image_file(F)
face_landmarks_list = face_recognition.face_landmarks(image)

def box_label(bgr, x1, y1, x2, y2, label): 
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 0, 0), 1, 1)
    cv2.rectangle(bgr, (int(x1), int(y1-25)), (x2, y1), (255,255,255), -1)
    cv2.putText(bgr, label, (x1, int(y1-5)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)

bgr = cv2.imread(F)
for i in face_landmarks_list:
    for j in i['left_eye']:
        cv2.circle(bgr, (j[0], j[1]), 2, (0, 255, 0), -1)
    for j in i['left_eyebrow']:
        cv2.circle(bgr, (j[0], j[1]), 2, (0, 0, 255), -1)


cv2.imshow('', bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

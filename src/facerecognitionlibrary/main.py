import cv2
import numpy as np
import face_recognition

imageElon = face_recognition.load_image_file('../../images/Elon Musk.jpeg')
imageElon = cv2.cvtColor(imageElon, cv2.COLOR_BGR2RGB)
imageTest = face_recognition.load_image_file('../../images/Elon Musk Test.jpeg')
imageTest = cv2.cvtColor(imageTest, cv2.COLOR_BGR2RGB)

# finding the faces in the image and the encoding

faceLoc = face_recognition.face_locations(imageElon)[0]
encodeElon = face_recognition.face_encodings(imageElon)[0]
cv2.rectangle(imageElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imageTest)[0]
encodeElonTest = face_recognition.face_encodings(imageTest)[0]
cv2.rectangle(imageTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

## Comparing these faces and finf=ding the distnce between them
results = face_recognition.compare_faces([encodeElon], encodeElonTest)
##print the result to see if faes are similar
# if you change the picture to say like the bill gates picture it does the encoding fine but results are false since the contours do not match
# Sometimes images may be similar so the best thing to do is to find similarities using distances the lower the distance the best the match
faceDistance = face_recognition.face_distance([encodeElon], encodeElonTest)
print(results, faceDistance)
cv2.putText(imageTest, f'{results} {round(faceDistance[0], 4)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Elon Musk", imageElon)
cv2.imshow("Elon Musk Test", imageTest)
cv2.waitKey(0)

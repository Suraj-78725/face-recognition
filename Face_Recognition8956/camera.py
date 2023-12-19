# import numpy as np
# import os
# import pickle
# import cv2
# import face_recognition
#
#
#
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
#
# imgBackground = cv2.imread("Resources/background.png")
# folderModePath = 'Resources/Modes'
# modePathList = os.listdir(folderModePath)
# imgModeList = []
# # import thing mode images into a list
# for path in modePathList:
#     imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
#
# # print(len(imgModeList))
# # load the face file
# print("Loading encoding file ...")
# file = open('Encodefile.p','rb')
# encodeListKnownWithids = pickle.load(file)
# file.close()
# encodeListKnown, studentIds = encodeListKnownWithids
# print(studentIds)
# print("Encoded file loaded")
# while True:
#     success, img = cap.read()
#
#     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#
#     faceCurFrame = face_recognition.face_locations(imgS)
#     encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
#
#     imgBackground[132:132 + 480, 70:70 + 640] = img  # for camera size
#     imgBackground[344:344 + 45, 850:850 + 270] = imgModeList[1]
#
#     for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.6)
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#         print("Facematches", matches)
#         print("FaceDis", faceDis)
#
#     cv2.imshow("webcam", img)
#     cv2.imshow("face Attendance", imgBackground)
#     cv2.waitKey(1)
#
#
#
# while True:
#     success, img = cap.read()
#
#     imgS = cv2.resize(img, (0, 0), None,0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#
#     faceCurFrame = face_recognition.face_locations(imgS)
#     encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
#
#     imgBackground[132:132 + 480, 70:70 + 640] = img  # for camera size
#     imgBackground[344:344 + 45, 850:850 + 270] = imgModeList[1]
#
#     for encodeFace,faceLoc in zip(encodeCurFrame, faceCurFrame):
#         # matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#         # faceDis = face_recognition.face_distance(encodeListKnown, encodeFace , np.array(encodeFace))
#         matches = face_recognition.compare_faces(np.array(encodeListKnown), np.array([encodeFace]), tolerance=0.6)
#         faceDis = face_recognition.face_distance(np.array(encodeListKnown), np.array([encodeFace]))
#
#         # matches = face_recognition.compare_faces(encodeListKnown, np.array(encodeFace), tolerance=0.6)
#         # faceDis = face_recognition.face_distance(np.array(encodeListKnown), np.array(encodeFace))
#         print("Facematches",matches)
#         print("FaceDis",faceDis)
#
#
#     cv2.imshow("webcam",img)
#
#
#     cv2.imshow("face Attendance", imgBackground)
#     cv2.waitKey(1)
# import cv2
# import os
# import pickle
# import face_recognition
# import numpy as np
import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("Resources/background.png")
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []

# Import mode images into a list
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Load the face file
print("Loading encoding file ...")
file = open('Encodefile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()

# Unpack the loaded data
encodeListKnown, studentIds = encodeListKnownWithIds

# Filter out elements with incorrect shapes
encodeListKnown = [np.array(encode) for encode in encodeListKnown if np.array(encode).shape == (128,)]

# Convert to NumPy array
encodeListKnown = np.array(encodeListKnown)

print(studentIds)
print("Encoded file loaded")

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[132:132 + 480, 70:70 + 640] = img  # for camera size
    imgBackground[344:344 + 45, 850:850 + 270] = imgModeList[1]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        encodeFace = np.array(encodeFace)  # Convert to NumPy array
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.6)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("Facematches", matches)
        # print("FaceDis", faceDis)
        matchIndex = np.argmin(faceDis)
        # print("Match Index ", matchIndex)

        if matches[matchIndex]:
            # print("Known face is detected")
            # print(studentIds[matchIndex])
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox =80+x1, 110+y1, x2-x1,y2-y1
            imgBackground =  cvzone.cornerRect(imgBackground, bbox, rt=0)

    cv2.imshow("webcam", img)
    cv2.imshow("face Attendance", imgBackground)
    cv2.waitKey(1)

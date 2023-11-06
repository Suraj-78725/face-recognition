import os

import cv2

cap =cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4,480)

imgBackground = cv2.imread("Resources/background.png")
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList= []
# import thing mode images into a list
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

# print(len(imgModeList))
while True:
    success, img =cap.read()
    imgBackground[132:132 + 480, 70:70 + 640] = img  #for camera size
    imgBackground[344:344 + 45, 850:850 + 270] = imgModeList[2]
    # cv2.imshow("webcam",img)
    cv2.imshow("face Attendance",imgBackground)
    cv2.waitKey(1)


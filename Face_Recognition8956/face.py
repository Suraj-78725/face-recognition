import cv2
import face_recognition
import pickle
import os

# importing student images
folderPath = 'images'
PathList = os.listdir(folderPath)
print(PathList)
imgList = []
studentIds = []
# import thing mode images into a list
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])
    # print(path)
    # print(os.path.splitext(path)[0])

print(studentIds)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if len(encode) > 0:
            encodeList.append(encode[0])
        else:
            # Handle the case when no faces are detected in the image
            encodeList.append(None)
        encodeList.append(encode)
    return encodeList


print("Encodeing Startingg ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithids = [encodeListKnown,studentIds]
# print(encodeListKnown)
print("Encodeing ending ...")


file = open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithids,file)
file.close()
print("file saved")
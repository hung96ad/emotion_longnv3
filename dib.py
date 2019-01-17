import cv2
import dlib
import numpy as np
import math
# import random
# import itertools
img_path='data/test_images/AF25ANS.JPG' # THE PATH OF THE IMAGE TO BE ANALYZED
font=cv2.FONT_HERSHEY_DUPLEX
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)) # Histogram equalization object
face_det=dlib.get_frontal_face_detector()
land_pred=dlib.shape_predictor("lib/DlibPredictor/shape_predictor_68_face_landmarks.dat")

def get_emotion():
    emotions = ["anger", "happy", "disgust","neutral","sadness","surprise"]
    return emotions
def crop_face(i_path):
    image=cv2.imread(i_path)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    face1 = cv2.CascadeClassifier("lib/HAARCascades/haarcascade_frontalface_default.xml")
    face2 = cv2.CascadeClassifier("lib/HAARCascades/haarcascade_frontalface_alt2.xml")
    face3 = cv2.CascadeClassifier("lib/HAARCascades/haarcascade_frontalface_alt.xml")
    face4 = cv2.CascadeClassifier("lib/HAARCascades/haarcascade_frontalface_alt_tree.xml")
    face_1 = face1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)
    face_2 = face2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)
    face_3 = face3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)
    face_4 = face4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face_1)==1:
        req_face=face_1
    elif len(face_2) == 1:
        req_face = face_2
    elif len(face_3) == 1:
        req_face = face_3
    elif len(face_4) == 1:
        req_face = face_4
    else:
        req_face=""
    if len(req_face)==1:
        for (x, y, w, h) in req_face:
            roi_gray = gray[y:y + h, x:x + w]
    else:
         print("\nFace not cropped using HAAR Cascade\n")
    img = cv2.resize(roi_gray, (350, 350))
    return img
def crop_face_to_des(i_path, des_path):
    final_name='error'
    image=cv2.imread(i_path)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    face1 = cv2.CascadeClassifier("lib/HAARCascades/haarcascade_frontalface_default.xml")
    face2 = cv2.CascadeClassifier("lib/HAARCascades/haarcascade_frontalface_alt2.xml")
    face3 = cv2.CascadeClassifier("lib/HAARCascades/haarcascade_frontalface_alt.xml")
    face4 = cv2.CascadeClassifier("lib/HAARCascades/haarcascade_frontalface_alt_tree.xml")
    face_1 = face1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)
    face_2 = face2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)
    face_3 = face3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)
    face_4 = face4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face_1)>0:
        req_face=face_1
    elif len(face_2) >0:
        req_face = face_2
    elif len(face_3) >0:
        req_face = face_3
    elif len(face_4) >0:
        req_face = face_4
    else:
        req_face=""
    if len(req_face)>0:
        for (x, y, w, h) in req_face:
            roi_gray = gray[y:y + h, x:x + w]
        temp_1 = i_path
        temp_split = temp_1.split('/')
        final_name = temp_split[-1]
        cv2.imwrite(des_path + '/' + final_name, cv2.resize(roi_gray, (350, 350)))
    else:
         print("\n Face not cropped using HAAR Cascade\n")
    print(final_name)
    return des_path + '/' + final_name
def get_landmarks(image_p):
    face_detections=face_det(image_p,1)
    for k,d in enumerate(face_detections):
        shape=land_pred(image_p,d)
        x_cords=[]
        y_cords=[]
        for i in range(1,68):
            x_cords.append(float(shape.part(i).x))
            y_cords.append(float(shape.part(i).y))
        xmean=np.mean(x_cords)
        ymean=np.mean(y_cords)
        x_central=[(x-xmean) for x in x_cords] # To compensate for variation in location of face in the frame.
        y_central=[(y-ymean) for y in y_cords]

        if x_cords[28]==x_cords[31]: # 26 is the top of the bridge, 29 is the tip of the nose
            anglenose=0
        else:
            anglenose_rad=int(math.atan((y_central[28] - y_central[31]) / (x_central[28] - x_central[31])))
            # Tan Inverse of slope
            anglenose=int(math.degrees(anglenose_rad))
        if anglenose<0:
            anglenose+=90      # Because anglenose computed above is the angle wrt to vertical
        else:
            anglenose-=90      # Because anglenose computed above is the angle wrt to vertical

        landmarks_v=[]
        for x,y,w,z in zip(x_central,y_central,x_cords,y_cords):
            landmarks_v.append(x) # Co-ordinates are added relative to the Centre of gravity of face to accompany for
            landmarks_v.append(y) # variation of location of face in the image.
            # # Euclidean distance between each point and the centre point (length of vector)
            # np_mean_coor=np.asarray((ymean,xmean))
            # np_coor=np.asarray((z,w))
            # euclid_d=np.linalg.norm(np_coor-np_mean_coor)
            # landmarks_v.append(euclid_d)

            # # Angle of the vector, which is used to correct for the offset caused due to tilt of image wrt horizontal
            # angle_rad = (math.atan((z - ymean) / (w - xmean)))
            # angle_degree = math.degrees(angle_rad)
            # angle_req = int(angle_degree - anglenose)
            # landmarks_v.append(angle_req)

    if len(face_detections)<1:
        landmarks_v="error"

    return  landmarks_v


    
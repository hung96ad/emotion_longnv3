import os
import sys
import logging

import numpy as np

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from dib import *

from utils.inference import detect_faces
from utils.inference import apply_offsets
from utils.inference import load_detection_model

detection_model_path = 'lib/HAARCascades/haarcascade_frontalface_default.xml'
SUPPORT_VECTOR_MACHINE_clf2 = joblib.load('Model/SVM_model_2.pkl').best_estimator_
emotions = ["anger", "happy", "disgust","neutral","sadness","surprise"]
face_detection = load_detection_model(detection_model_path)
emotion_offsets = (0, 0)
def process_image(image):
    results = []
    try:
        # loading images
        image_array = np.fromstring(image, np.uint8)
        unchanged_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

        rgb_image = cv2.cvtColor(unchanged_image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(unchanged_image, cv2.COLOR_BGR2GRAY)

        faces = detect_faces(face_detection, gray_image)
        for face_coordinates in faces:
            pred_data=[]
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                clahe_gray=clahe.apply(gray_face)
                landmarks_vec = get_landmarks(clahe_gray)
                pred_data.append(landmarks_vec)
                if landmarks_vec == "error":
                    pass
                else:
                    pred_data.append(landmarks_vec)
                    np_test_data = np.array(pred_data)
                    emotion = SUPPORT_VECTOR_MACHINE_clf2.predict(pred_data)
                    results.append(emotions[emotion[0]])
            except:
                continue
    except Exception as err:
        logging.error('Error in emotion processor: "{0}"'.format(err))
    return results
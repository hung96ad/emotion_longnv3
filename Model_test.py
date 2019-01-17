import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from dib import *
des_path='data/cap_image'
img_path='data/test_images/happy2.jpg' 
font = cv2.FONT_HERSHEY_DUPLEX
emotions = get_emotion()
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)) # Histogram equalization object

SUPPORT_VECTOR_MACHINE_clf2 = joblib.load('Model/SVM_model_2.pkl').best_estimator_

pred_data=[]
pred_labels=[]
a_path = crop_face_to_des(img_path, des_path)
img = cv2.imread(a_path)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe_gray=clahe.apply(gray)
landmarks_vec = get_landmarks(clahe_gray)
if landmarks_vec == "error":
    pass
else:
    pred_data.append(landmarks_vec)
    np_test_data = np.array(pred_data)
    print(len(np_test_data[0]))
    a=SUPPORT_VECTOR_MACHINE_clf2.predict(pred_data)
    # cv2.putText(img,'EMOTION: ',(8,30),font,0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(img,emotions[a[0]].upper(),(150,60),font,1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('test_image',img)
    cv2.imwrite('data/test_images/a2.jpg', img);
    print(emotions[a[0]])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

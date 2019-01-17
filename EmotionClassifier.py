import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import glob
import random
import os
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from dib import get_landmarks, get_emotion
emotions = get_emotion()
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)) # Histogram equalization object

def get_images(m_path,emotion):
    i_path=glob.glob(m_path+emotion+'/*')
    random.shuffle(i_path) # random ima path
    train_paths=i_path[:int(len(i_path)*0.80)] 
    predict_paths=i_path[-int(len(i_path)*0.20):]
    return train_paths,predict_paths
def org_data(m_path):
    train_data=[]
    train_labels=[]
    pred_data=[]
    pred_labels=[]
   
    for emo in emotions:
        train_p,pred_p=get_images(m_path, emo)
        for im in train_p:
            img=cv2.imread(im)
            img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            clahe_gray=clahe.apply(img_gray)
            landmarks_vec=get_landmarks(clahe_gray)
            if landmarks_vec == "error":
                pass
            else:
                train_data.append(landmarks_vec)
                train_labels.append(emotions.index(emo))
        for im in pred_p:
            img=cv2.imread(im)
            img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            clahe_gray=clahe.apply(img_gray)
            landmarks_vec=get_landmarks(clahe_gray)
            if landmarks_vec == "error":
                pass
            else:
                pred_data.append(landmarks_vec)
                pred_labels.append(emotions.index(emo))
    return np.array(train_data),np.array(train_labels),np.array(pred_data),np.array(pred_labels)


pathmd='Model/softmax_model_main.pkl'
path_data='data/FaceDB/OrganizedData_2/'
train_data, train_labels, pred_data, pred_labels=org_data(path_data)
if os.path.exists(pathmd):
    print('Using model from file:' + pathmd)
    clf = joblib.load(pathmd).best_estimator_
else:
    print('Training model.')
    # params_grid = {'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5,1], 'C': [1, 1e3,2e1, 5e3,1e5, 2e5, 3e5]}
    logreg = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')

    params_grid = {'C': [1,1e3,2e1, 5e3,1e5, 2e5, 3e5]}
    # clf = GridSearchCV(SVC(kernel='poly', class_weight='balanced'), param_grid, refit=True)      
    # params_grid = {'hidden_layer_sizes': [(10,), (20,), (50,), (128, 256, 128,)]}
    mlp = MLPClassifier(verbose=10, learning_rate='adaptive')  
    clf = GridSearchCV(logreg, params_grid, refit=True, n_jobs=-1, cv=5)
    clf.fit(train_data, train_labels)
    print('Finished with grid search with best mean cross-validated score:', clf.best_score_)
    print('Best params appeared to be', clf.best_params_)
    joblib.dump(clf, pathmd)
np_test_data=pred_data
print('Test accuracy:', clf.score(np_test_data, pred_labels))






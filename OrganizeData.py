import glob
from shutil import copyfile
from dib import get_emotion
import os
emotions = get_emotion()
candidates_paths=glob.glob('FaceDB/cohn-kanade-images/*')
path_2='FaceDB/FaceDB2/'
# for x in candidates_paths:
#     serial=x[-4:]
#     for sessions in glob.glob("FaceDB/Emotion/%s/*" %serial):
#         for files in glob.glob("%s/*" %sessions):       
#             file = open(files,'r')
#             current_session = str(files).split('/')[-2]
#             current_emotion = int(float(file.readline()))  
#             a=glob.glob("FaceDB/cohn-kanade-images/%s/%s/*" %(serial, current_session))
#             b=sorted(a)
#             image_emotion_s = b[-1]
#             image_neutral_s = b[0]  
#             neutral_d="data/FaceDB/OrganizedData/neutral/%s" %str(image_neutral_s).split('/')[-1]
#             emo_d = "data/FaceDB/OrganizedData/%s/%s" %(emotions[current_emotion],str(image_emotion_s).split('/')[-1])
#             copyfile(image_neutral_s,neutral_d)
#             copyfile(image_emotion_s,emo_d)
def search_face():
    emotions_ha={'AF':'afraid','AN':'anger', 'DI':'disgust' ,'HA':'happy', 'NE':'neutral', 'SA':'sadness', 'SU':'surprise'}
    paths = os.listdir(path_2)
    for j in paths:
        for emo_key in emotions_ha.keys():
            fn = path_2 + str(j)+'/'+str(j)+ str(emo_key).zfill(2) + 'S' + '.JPG'
            emo_d = "data/FaceDB/OrganizedData_2/%s/%s" %(emotions_ha[emo_key],str(fn).split('/')[-1])
            copyfile(fn,emo_d)
       

search_face()





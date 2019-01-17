import glob
import cv2
from dib import crop_face
m_dir_2='data/FaceDB/OrganizedData_2/'
m_dir='data/FaceDB/OrganizedData/'
dest_m_dir_2='data/FaceDB/Crop_Organized_Data_2/'
dest_m_dir='data/FaceDB/Crop_Organized_Data/'

for emo_path in glob.glob(m_dir + '*'):
    for image_path in glob.glob(emo_path+'/*'):
        temp_1=image_path
        temp_split=temp_1.split('/')
        final_name=temp_split[-1]
        cv2.imwrite(dest_m_dir+str(emo_path).split('/')[-1]+'/'+final_name,crop_face(image_path))
for emo_path in glob.glob(m_dir_2 + '*'):
    for image_path in glob.glob(emo_path+'/*'):
        temp_1=image_path
        temp_split=temp_1.split('/')
        final_name=temp_split[-1]
        cv2.imwrite(dest_m_dir_2+str(emo_path).split('/')[-1]+'/'+final_name,crop_face(image_path))
        





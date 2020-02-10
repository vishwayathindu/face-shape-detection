import os
import cv2
import dlib
from imutils import face_utils  

datapath='face shapes'

labels=os.listdir(datapath)
# use (os.mkdir('myFolder/test.txt')) to create a folder

label_dict={}

for i in range(len(labels)):

    label_dict[labels[i]]=i

print(label_dict)

face_detector=dlib.get_frontal_face_detector()
#loading the pre trained algorithm for 68 landmark  face detection available in dlib

landmark_detector=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#loading the pretrained algorithm for detection available in dlib

data=[]
target=[]

def measure_distances(img,label):


    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##    converting the color

    rects=face_detector(gray)
    #same like ::algorithm.predict(test_data)

    for rect in rects:
        x1=rect.left()
        y1=rect.top()
        x2=rect.right()
        y2=rect.bottom()

        
        #cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)

        points=landmark_detector(gray,rect)
        #passing the gray image and ROI (detected face rectangle) to the landmark_detector
        #return is 68 points(x,y) (note-this is not a numpyy array)

        points=face_utils.shape_to_np(points)

        myPoints=points[2:9,0]
        #getting the x cordinates of point 2-8

        D1=myPoints[6]-myPoints[0]
        D2=myPoints[6]-myPoints[1]
        D3=myPoints[6]-myPoints[2]
        D4=myPoints[6]-myPoints[2]
        D5=myPoints[6]-myPoints[4]
        D6=myPoints[6]-myPoints[5]

        d1=D2/D1
        d2=D3/D1
        d3=D4/D1
        d4=D5/D1
        d5=D6/D1

        data.append([d1,d2,d3,d4,d5])
        
        target.append(label)
        
        

    

for label in labels:

    imgs_path=os.path.join(datapath,label)
    img_names=os.listdir(imgs_path)

    for img_name in img_names:
    
        img_path=os.path.join(imgs_path,img_name)
        img=cv2.imread(img_path)

        cv2.imshow('live',img)
        cv2.waitKey(100)

        measure_distances(img,label_dict[label])

import numpy as np

np.save('data',data)
np.save('target',target)
        
print(data)
print(target)

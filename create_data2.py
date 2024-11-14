import os
import cv2
directory = "project"
os.makedirs(directory,exist_ok=True)
print( "created successfully.")
#creating database
#it captures images and stores them in datasets
#folder under the folder name of sub data
import cv2, sys,numpy,os
haar_file = 'haarcascade_frontalface_default.xml'
#all the faces 
#data will be present in this folder
datasets = 'datasets'
#these are sub data sets of my folder
#for my faces 
#I've used my name 'DIVI'
#u can even change it 
sub_data = 'divi'
path = os.path.join(directory, datasets, sub_data)
if not os.path.isdir(path):
    os.makedirs(path)
#defining the size of images
(width,height) = (130,100)
#'0' is used for my webcam
#if u've any other camera attached
#use '1' like this
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(1)
#the program loops until has 4 mages of face
count =1
while count<=30:
    (ret, im) = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face,(width,height))
        cv2.imwrite(f'{path}/{count}.png', face_resize)
    count+=1
    cv2.imshow('OpenCV',im)
    key = cv2.waitKey(10)
    if key==27:  # Press 'ESC' to exit the loop
        break
webcam.release()
cv2.destroyAllWindows()

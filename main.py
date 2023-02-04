import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
#  import pygame as pg
#from pygame import mixer


face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

model = load_model(r"C:\Users\Saksham Hans\Desktop\Driver Drowsiness Project\models\model.h5")

#mixer.init()
#sound = mixer.sound(r'C:\Users\Saksham Hans\PycharmProjects\Driver Drowsiness\alarm.wav')
cap = cv2.VideoCapture(0)
score = 0
while True:
    ret, frame = cap.read()
    height,width = frame.shape[0:2]
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2)

    cv2.rectangle(frame,(0,height-50),(200,height),(0,0,0),thickness=cv2.FILLED)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h), color=(255,0,0),thickness=3)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color=(255,0,0),thickness=3)

        #preprocessing steps
        eye = frame[ey:ey+eh,ex:ex+ew]
        eye = cv2.resize(eye,(80,80))
        eye = eye/255
        eye = eye.reshape(80,80,3)
        eye = np.expand_dims(eye,axis=0)

        #model prediction
        prediction = model.predict(eye)
        if prediction[0][0] > 0.50:
            cv2.putText(frame,'Closed',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),thickness=1,lineType=cv2.LINE_AA)
            cv2.putText(frame, 'Score:' + str(score), (100, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                        color=(255, 255, 255), thickness=1)
            score = score +1;
            if score > 15:
                try:
                    pass
                    #sound.play()
                except:
                    pass
        elif prediction[0][1] > 0.60:
            cv2.putText(frame,'Opened',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),thickness=1,lineType=cv2.LINE_AA)
            cv2.putText(frame, 'Score:' + str(score), (100, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=1,
                        color=(255, 255, 255), thickness=1)
            score = score -1
            if score < 0:
                score =0

        #print(prediction)

    cv2.imshow('frame',frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('frame')

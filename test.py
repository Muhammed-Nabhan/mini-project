import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


cap = cv2.VideoCapture('http://100.86.147.26:8080/video')
detector=HandDetector(maxHands=1)
classifier=Classifier("model/keras_model.h5","model/labels.txt")
offset=20
imgSize=300
folder="data/C"
counter=0
labels=["A","B","C"]
while(cap.isOpened()):
    ret,frame=cap.read()
    imgOutput=frame.copy()
    hands,img=detector.findHands(frame)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']

        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop=frame[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropShape=imgCrop.shape


        aspectRatio=h/w
        if aspectRatio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap]=imgResize
            prediction,index=classifier.getPrediction(imgWhite)
            print(prediction,index)


        else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:]=imgResize
            prediction,index=classifier.getPrediction(imgWhite)

        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
            
        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)

        cv2.imshow("Image",imgOutput)
     
    try:
        cv2.imshow('temp',cv2.resize(frame, (600,400)))
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
    except cv2.error:
        print("Stream ended...")
        break
        
cap.release()
cv2.destroyAllWindows()
from flask import Flask,render_template,Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

app = Flask(__name__)


cap = cv2.VideoCapture(2)
detector=HandDetector(maxHands=1)
classifier=Classifier("model/keras_model.h5","model/labels.txt")
offset=20
imgSize=300
folder="data/C"
counter=0
labels=["A","B","C"]

def generate_frames():
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
                    
            ret, buffer =cv2.imencode('.jpg',imgOutput)
            frame_bytes=buffer.tobytes()

            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +frame_bytes + b'\r\n\r\n')
@app.route('/')
def index():
        return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')
if __name__ == '__main__':
    app.run(debug=True)
                #cv2.imshow("ImageCrop",imgCrop)
                #cv2.imshow("ImageWhite",imgWhite)

                #cv2.imshow("Image",imgOutput)
            
            ##try:
              ##  cv2.imshow('temp',cv2.resize(frame, (600,400)))
                ##key=cv2.waitKey(1)
                ##if key == ord('q'):
                  ##  break
            ##except cv2.error:
              ##  print("Stream ended...")
               ## break
            
##cap.release()
##cv2.destroyAllWindows()
'''
FACE DETECTION AND TRACKING PROJECT
By using : Haar Cascade Frontal Face Algorithm
The accuracy of this project may be low
can be used in projects such as : selfie camera...

'''
import cv2 as cv
import imutils as im
import time as t
import os


cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
alg = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')


#alg = "haarcascade_frontalface_default.xml" #accessed the model
haarcasc = cv.CascadeClassifier(alg) # loading the model

cam = cv.VideoCapture(0)
t.sleep(1)

firstframe =None
area = 500


while True:
    _,img = cam.read()
    grayimg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face = haarcasc.detectMultiScale(grayimg,1.3,4)  #get the coordinates of the face
    for (x,y,w,h) in face:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv.imshow("face detection",img)
    key = cv.waitKey(10)
    if key ==27: # clicking the esc button
            break
cam.release()
cv.destroyAllWindows()  

#!/usr/bin/python3
import cv2

# calling classifire
casclf=cv2.CascadeClassifier('face.xml')
casclf1=cv2.CascadeClassifier('eye.xml')
# loading face data
cap=cv2.VideoCapture(0)

while cap.isOpened():
    status,frame=cap.read()
    print(frame)
    # now we apply classifier in lin=ve frame
    face=casclf.detectMultiScale(frame,1.5,5) #classifier turing parameter
    eye=casclf1.detectMultiScale(frame,1.5,5) #classifier turing parameter
    #print(face)
    for x,y,h,w in face:
        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),3)
        # only face
        facedata=frame[x:x+h,y:y+w]
        
        for x,y,h,w in eye:
           cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),3)
           eyedata=frame[x:x+h,y:y+w]

    cv2.imshow('face',frame)
    if cv2.waitKey(10)  &  0xff == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

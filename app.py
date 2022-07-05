from cv2 import cv2
import os
import sys
from tensorflow import keras
from keras.models import load_model
import numpy as np
from pygame import mixer
import multiprocessing
from playsound import playsound
import time
import tkinter as tk

e = multiprocessing.Event()
p = None


if getattr(sys, 'frozen', False):
    THIS_FOLDER = os.path.dirname(sys.executable)
elif __file__:
    THIS_FOLDER = os.path.dirname(__file__)

capturing = False
def startCapture():
    global capturing
    capturing = not capturing



model = load_model(os.path.join(THIS_FOLDER, 'cnnBasic.h5'))

faceCascade = cv2.CascadeClassifier(os.path.join(THIS_FOLDER, 'haarcascade_frontalface_default.xml'))
eye_cascade_main = cv2.CascadeClassifier(os.path.join(THIS_FOLDER, 'haarcascade_eye.xml'))
eye_cascade_backup = cv2.CascadeClassifier(os.path.join(THIS_FOLDER, 'haarcascade_righteye_2splits.xml'))

img_size = 128

def preprocessing(frame, roi_gray,roi_color, score):
    eye_cascade = eye_cascade_main
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5)
    
    #change eye_cascade if couldn't detcet eye with current eye_cascade
    if(len(eyes)==0):
        eye_cascade = eye_cascade_backup
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5)
        
    for (ex,ey,ew,eh) in eyes: #green
        eye = roi_color[ey:ey+eh, ex:ex+ew]
        eye = cv2.resize(eye, (img_size, img_size))

        cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh),(0,255,0), 2)
        eye = np.expand_dims(eye, axis=0)
        eye= eye / 255
        
        score = prediction(frame, eye, score)
        break
        
    return score

def prediction(frame, eye, score):
    prediction = model.predict(eye)
    if(prediction[0][0]> 0.5):
        cv2.putText(frame, 'Open', (0, -10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return False
    else:
        cv2.putText(frame, 'Close', (0, -10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2) 
        return True

from datetime import datetime
capturing = True
cap = cv2.VideoCapture(0)
def captureVideo():
    global capturing
    global cap
    
    score = 0
    prev = datetime.now()
    while capturing:
        cur_time = datetime.now()
        ret, frame = cap.read()
        time_elapsed = (cur_time - prev).total_seconds() * 1000
        # add GaussianBlur to remove noises
        frame = cv2.GaussianBlur(frame, (1, 1), 0)
        height,width = frame.shape[:2] 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, minNeighbors=5,scaleFactor=1.1)
        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            score = preprocessing(frame, roi_gray, roi_color, score)
            if not score:
                prev = datetime.now()
            if time_elapsed > 650 and score:
                print("sleep")
                prev = datetime.now()
                score = False
                playsound(os.path.join(THIS_FOLDER, 'alarm.wav'))
            break

        cv2.imshow('h', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
          cap.release()
          cv2.destroyAllWindows()
          break

def endCapturing():
    global capturing
    global cap
    e.set()
    p.join()
    capturing = False
    cap.release()
    cv2.destroyAllWindows()

def start_recording_proc():
    global p
    p = multiprocessing.Process(target=captureVideo, args=(e,))
    p.start()

main = tk.Tk()
ws = main.winfo_screenwidth() 
hs = main.winfo_screenheight()
w = int(ws*0.7)
h = int(hs*0.7)
main.geometry("{}x{}".format(w,h)) #set window size
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
main.geometry('+%d+%d' % (x, y))

start_btn = tk.Button(main, text="َشروع", command=start_recording_proc)
start_btn.pack()

end_btn = tk.Button(main, text="پایان", command=endCapturing)
end_btn.pack()

main.mainloop()
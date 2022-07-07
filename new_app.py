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
from tkinter import filedialog as fd
import PIL.Image, PIL.ImageTk
from datetime import datetime


if getattr(sys, 'frozen', False):
    THIS_FOLDER = os.path.dirname(sys.executable)
elif __file__:
    THIS_FOLDER = os.path.dirname(__file__)

class App:
    def __init__(self, window, window_title, video_source=0):
        BGCOLOR = '#ccf5ff'
        self.window = window
        self.window.configure(bg=BGCOLOR)
        self.alarmFilePath = ''
        ws = self.window.winfo_screenwidth() 
        hs = self.window.winfo_screenheight()
        w = int(ws)
        h = int(hs * 0.9)
        self.window.geometry("{}x{}".format(w,h)) #set window size
        x = (ws/2) - (w/2)
        self.window.geometry('+%d+%d' % (x, 0))
        self.window.title(window_title)
        self.video_source = video_source
        self.ok = False

        # open video source (by default this will try to open the computer webcam)
        self.videoCap = VideoCapture(self.video_source)

        # Radio buttons
        radio_Frame = tk.Frame(window, bg='#ccf5ff')
        radio_Frame.pack(side=tk.RIGHT, padx=10)
        values= ["600", "650", "700"]
        self.v = tk.IntVar(value=650)
        for val in values:
            tk.Radiobutton(radio_Frame, 
                text=val,
                bg=BGCOLOR,
                pady = 20, 
                variable=self.v,
                value=val).pack()
            # self.radio1.grid(row = 0, column = 0)

        alarm_Frame = tk.Frame(window, bg='#ccf5ff')
        alarm_Frame.pack(side=tk.LEFT, padx=5, anchor="e")
        # 
        tk.Label(
            alarm_Frame,
            text="در این قسمت می توانید زنگ هشدار مورد نظر خود", 
            bg='#ccf5ff').pack()
            
        tk.Label(
            alarm_Frame, 
            text="را انتخاب کنید. در صورتی که زنگی انتخاب نکنید", 
            bg='#ccf5ff').pack()

        tk.Label(
            alarm_Frame, 
            text="از زنگ هشدار پیش فرض استفاده خواهد شد", 
            bg='#ccf5ff').pack()

        tk.Button(
            alarm_Frame, 
            bg='#98e6e6', 
            text='انتخاب فایل', 
            bd=0,
            command=self.openFile, 
            height = 2, 
            width = 8
        ).pack(pady = 5)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width = w * 0.5, height = h * 0.7, bg=BGCOLOR)
        self.canvas.pack(side=tk.TOP)



        #
        start_cancel = tk.Frame(window, bg=BGCOLOR)
        start_cancel.pack(side='bottom')

        #video control buttons

        self.btn_start=tk.Button(start_cancel, bd=0, bg='#98e6e6', text='START', command=self.open_camera, height = 3, width = 7)
        self.btn_start.pack(side=tk.LEFT, padx=10)

        self.btn_stop=tk.Button(start_cancel,bd=0, bg='#98e6e6', text='STOP', command=self.close_camera, height = 3, width = 7)
        self.btn_stop.pack(side=tk.LEFT, padx=10)

        # quit button
        self.btn_quit=tk.Button(start_cancel,bd=0, bg='#98e6e6', text='QUIT', command=sys.exit, height = 3, width = 7)
        self.btn_quit.pack(side=tk.LEFT,padx=10)


        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay=10
        self.window.mainloop()


    def openFile(self):
        self.alarmFilePath = fd.askopenfilename()
    def open_camera(self):
        self.ok = True
        self.videoCap.prev = datetime.now()
        self.update()
        print("camera opened => Recording")


    def close_camera(self):
        self.ok = False
        print("camera closed => Not Recording")

       
    def update(self):
        # Get a frame from the video source
        if self.ok:
            ret, frame = self.videoCap.get_frame(self.v.get(), self.alarmFilePath)

            if ret:
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.window.after(self.delay, self.update)


class VideoCapture:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.score = False
        self.img_size = 128
        self.model = load_model(os.path.join(THIS_FOLDER, 'data/cnnBasic.h5'))
        self.faceCascade = cv2.CascadeClassifier(os.path.join(THIS_FOLDER, 'data/haarcascade_frontalface_default.xml'))
        self.eye_cascade_main = cv2.CascadeClassifier(os.path.join(THIS_FOLDER, 'data/haarcascade_eye.xml'))
        self.eye_cascade_backup = cv2.CascadeClassifier(os.path.join(THIS_FOLDER, 'data/haarcascade_righteye_2splits.xml'))
        self.prev = datetime.now()

    def preprocessing(self, frame, roi_gray,roi_color, score):
        eye_cascade = self.eye_cascade_main
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5)
    
    #change eye_cascade if couldn't detcet eye with current eye_cascade
        if(len(eyes)==0):
            eye_cascade = self.eye_cascade_backup
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5)

        for (ex,ey,ew,eh) in eyes: #green
            eye = roi_color[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (self.img_size, self.img_size))

            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh),(0,255,0), 2)
            eye = np.expand_dims(eye, axis=0)
            eye= eye / 255

            self.score = self.prediction(frame, eye, score)
            break
        
        return self.score

    def prediction(self, frame, eye, score):
        prediction = self.model.predict(eye)
        if(prediction[0][0]> 0.5):
            cv2.putText(frame, 'Open', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            return False
        else:
            cv2.putText(frame, 'Close', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2) 
            return True

        # To get frames
    def get_frame(self, threshold, alarmFilePath):
        if self.cap.isOpened():
            cur_time = datetime.now()
            ret, frame = self.cap.read()
            if ret:
                time_elapsed = (cur_time - self.prev).total_seconds() * 1000
                # add GaussianBlur to remove noises
                frame = cv2.GaussianBlur(frame, (1, 1), 0)
                height, width = frame.shape[:2] 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(gray, minNeighbors=5,scaleFactor=1.1)
                cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    self.score = self.preprocessing(frame, roi_gray, roi_color, self.score)
                    if not self.score:
                        self.prev = datetime.now()
                    if time_elapsed > threshold and self.score:
                        print("sleep")
                        self.prev = datetime.now()
                        self.score = False
                        if alarmFilePath:
                            playsound(alarmFilePath)
                        else:
                            playsound(os.path.join(THIS_FOLDER, 'data/alarm.wav'))

                    break
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                return (ret, None)
        else:
            return (ret, None)

        # Release the video source when the object is destroyed
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
            # self.out.release()
            cv2.destroyAllWindows()

def main():
    # Create a window and pass it to the Application object
    App(tk.Tk(),'Video Recorder')

main()
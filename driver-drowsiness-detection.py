from cv2 import cv2
import os
import sys
from tensorflow import keras
from keras.models import load_model
import numpy as np
from playsound import playsound
import time
import tkinter as tk
from tkinter import filedialog as fd
import PIL.Image, PIL.ImageTk
from datetime import datetime

# Find the current dir
if getattr(sys, 'frozen', False):
    THIS_FOLDER = os.path.dirname(sys.executable)
elif __file__:
    THIS_FOLDER = os.path.dirname(__file__)

## This class implements our graphic interface
class App:
    def __init__(self, window, window_title, video_source=0):
        BGCOLOR = '#ccf5ff'
        self.window = window
        self.window.configure(bg=BGCOLOR) #set window background
        self.alarmFilePath = ''
        ws = self.window.winfo_screenwidth() 
        hs = self.window.winfo_screenheight()
        w = int(ws)
        h = int(hs * 0.9)
        self.window.geometry("{}x{}".format(w,h)) #set window size
        x = (ws/2) - (w/2)
        self.window.geometry('+%d+%d' % (x, 0)) # set window horizentally and vertically in center
        self.window.title(window_title)
        self.video_source = video_source
        self.ok = False

        # Open video source (by default this will try to open the computer webcam)
        self.videoCap = VideoCapture(self.video_source)

        # Alarm threshold's panel
        radio_Frame = tk.Frame(window, bg='#ccf5ff')
        radio_Frame.pack(side=tk.RIGHT, padx=20, anchor=tk.E)
        tk.Label(
            radio_Frame, 
            text="می‌توانید حد آستانه زنگ هشدار را تغییر دهید", 
            bg='#ccf5ff').pack(side=tk.TOP, padx=5)

        # Radio buttons
        values= ["550", "600", "650"]
        self.v = tk.IntVar(value=600)
        for val in values:
            tk.Radiobutton(radio_Frame, 
                text=val,
                bg=BGCOLOR,
                variable=self.v,
                value=val).pack(side=tk.RIGHT, padx=5)

        # Alarm selection panel
        alarm_Frame = tk.Frame(window, bg='#ccf5ff')
        alarm_Frame.pack(side=tk.LEFT, padx=10)
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


        # Choose file button
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

        # Video controll panel
        start_cancel = tk.Frame(window, bg=BGCOLOR)
        start_cancel.pack(side='bottom')

        # Video control buttons

        self.btn_quit=tk.Button(start_cancel, bd=0, bg='#98e6e6', text='پایان', command=sys.exit, height = 3, width = 7)
        self.btn_quit.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_stop=tk.Button(start_cancel, bd=0, bg='#98e6e6', text='توقف', command=self.close_camera, height = 3, width = 7)
        self.btn_stop.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_start=tk.Button(start_cancel, bd=0, bg='#98e6e6', text='شروع', command=self.open_camera, height = 3, width = 7)
        self.btn_start.pack(side=tk.LEFT, padx=10, pady=10)


        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay=10
        self.window.mainloop() #start tkinter mainloop

    # Open file explorer
    def openFile(self):
        self.alarmFilePath = fd.askopenfilename()

    # Open camera
    def open_camera(self):
        self.ok = True
        self.videoCap.prev = datetime.now()
        self.update()

    # Close camera
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

# This class implements our logic
class VideoCapture:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.score = False
        self.img_size = 128
        self.model = load_model(os.path.join(THIS_FOLDER, 'data/cnnBasic.h5'))
        self.face_cascade = cv2.CascadeClassifier(os.path.join(THIS_FOLDER, 'data/haarcascade_frontalface_default.xml'))
        self.right_eye_cascade = cv2.CascadeClassifier(os.path.join(THIS_FOLDER, 'data/haarcascade_righteye_2splits.xml'))
        self.left_eye_cascade = cv2.CascadeClassifier(os.path.join(THIS_FOLDER, 'data/haarcascade_lefteye_2splits.xml'))
        self.prev = datetime.now()

    # Extract eyes and resize them, then send it to predict function
    def preprocessing(self, frame, roi_face_gray, roi_face_rgb):
        eye_cascade = self.right_eye_cascade
        eyes = eye_cascade.detectMultiScale(roi_face_gray, scaleFactor=1.1, minNeighbors=5)
    
        # Change eye_cascade if couldn't detcet eye with current eye_cascade
        if(len(eyes)==0):
            eye_cascade = self.left_eye_cascade
            eyes = eye_cascade.detectMultiScale(roi_face_gray, scaleFactor=1.1, minNeighbors=5)

        for (ex,ey,ew,eh) in eyes: 
            eye = roi_face_rgb[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (self.img_size, self.img_size)) #resize eye image

            cv2.rectangle(roi_face_rgb,(ex, ey),(ex+ew, ey+eh),(0,255,0), 2) #green rectangle for eye
            eye = np.expand_dims(eye, axis=0)
            eye= eye / 255 # normalize array
            self.prediction(frame, eye)
            break
    
    # Get eye as an arg and classifys it
    def prediction(self, frame, eye):
        prediction = self.model.predict(eye)
        if(prediction[0][0]> 0.5):
            cv2.putText(frame, 'Open', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            self.score = False
        else:
            cv2.putText(frame, 'Close', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2) 
            self.score = True

    ##Capture frames and send face region to preprocessing method
    ## then, check drowsiness and plays an alarm
    def get_frame(self, threshold, alarmFilePath):
        if self.cap.isOpened():
            cur_time = datetime.now()
            ret, frame = self.cap.read()
            if ret:
                time_elapsed = (cur_time - self.prev).total_seconds() * 1000
                # add GaussianBlur to remove noises
                frame = cv2.GaussianBlur(frame, (1, 1), 0)
                height, _ = frame.shape[:2] 

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert image to gray for detecting face
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) 
                cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    self.preprocessing(frame, roi_gray, roi_color)
                    
                    # check drowsiness
                    if not self.score:
                        self.prev = datetime.now()
                        
                    if time_elapsed > threshold and self.score:
                        # plays alarm
                        if alarmFilePath: 
                            playsound(alarmFilePath)
                        else:
                            playsound(os.path.join(THIS_FOLDER, 'data/alarm.wav'))

                        self.score = False
                        self.prev = datetime.now()


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
            cv2.destroyAllWindows()

def main():
    # Create a window and pass it to the Application object
    App(tk.Tk(),'تشخیص خواب‌آلودگی رانندگان')

main()
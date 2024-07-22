# Driver-Drowsiness-Detection
This is a system which can detect the drowsiness of the driver using CNN - Python, OpenCV

The aim of this is system to reduce the number of accidents on the road by detecting the drowsiness of the driver and warning them using an alarm.

Here, we used Python, OpenCV, Keras(tensorflow) to build a system that can detect features from the face of the drivers and alert them if ever they fall asleep while while driving. The system dectects the eyes and prompts if it is closed or open.
If the eyes are closed for 600 miliseconds it will play the alarm to get the driver's attention, to stop cause its drowsy.
We have build a CNN network which is trained on MRL dataset which can detect closed and open eyes. 
Then OpenCV is used to get the live fed from the camera and run that frame through the CNN model to process it and classify wheather it opened or closed eyes.

## setup
Pre-install all the required libraries
1) OpenCV
2) Keras
3) Numpy
4) Tkinter
5) OS
6) playsound

Download/Clone this repository
Inside the main folder, open a terminal and run => python driver-drowsiness-detection.py

## The dataset
The dataset which was used is a subnet of a dataset from [MRL](http://mrl.cs.vsb.cz/eyedataset)
We used 14500 images for training and testing

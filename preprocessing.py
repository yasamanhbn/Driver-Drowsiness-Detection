from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

train_dir = './base'
Classes = ['closed', 'open']
img_size = 128

def create_training_data():
    training_data = []
    for category in Classes:
        path = os.path.join(train_dir, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
            backtorgb  = cv2.resize(backtorgb, (img_size,img_size))
            training_data.append([backtorgb, class_num])
    return training_data

def loadData():
    X_train = []
    y_label = []
    training_data = create_training_data()
    random.shuffle(training_data)

    for features, label in training_data:
        X_train.append(features)
        y_label.append(label)

    X_train = np.array(X_train)
    X_train = X_train/255.0
    y_label = np.array(y_label)

    return train_test_split(X_train, y_label, test_size=0.2, random_state=42)

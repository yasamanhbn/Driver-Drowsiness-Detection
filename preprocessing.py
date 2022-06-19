
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator


# Reading all images from dataset and labeling them


def create_training_data():
    train_dir = '../dataset/base'
    Classes = ['closed', 'open']
    training_data = []
    for category in Classes:
        path = os.path.join(train_dir, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try :
                img_array = cv2.imread(os.path.join(path,img))
                training_data.append([img_array, class_num])
            except Exception as e:
                print(e)

def loadData():
    train_dir = '../dataset/base'
    img_size = 224
    batch_size = 64
    create_training_data()
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    #test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,      # This is the target directory
        target_size=(img_size, img_size),    # All images will be resized to "img_size" * "img_size"
        batch_size = batch_size,
        class_mode='binary',           # Since we use binary_crossentropy loss, we need binary labels
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        train_dir, # same directory as training data
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation') # set as validation data

    return (train_generator, validation_generator)


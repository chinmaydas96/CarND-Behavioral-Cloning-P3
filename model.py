## Import the necessary libraries

import os
import csv
import cv2
import sklearn
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPool2D, Cropping2D, Flatten, Dense,Dropout 
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random


## Read the csv file for extraction of path of the image

lines = []  
with open('./new_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader :
        lines.append(line)

## Spliting dataset to trainset and validation set
training_data, validation_data = train_test_split(lines,test_size=0.2)


def make_dataset(line, images, measurements):
    """
    This function reads data from line and adds it to the list of input images
    and measurements
    """

    # correction for left and right camera image
    steering_angle_correction = 0.2

    # load images and measurements
    
    "./new_data/" + line[0].replace('/Users/bat/image_data/','')
    image_center = cv2.imread("./new_data/" + line[0].replace('/Users/bat/image_data/',''))
    image_left = cv2.imread("./new_data/" + line[0].replace('/Users/bat/image_data/',''))
    image_right = cv2.imread("./new_data/" + line[0].replace('/Users/bat/image_data/',''))
    steering_angle = float(line[3])
    # correct angles for left and right image
    steering_angle_left = steering_angle + steering_angle_correction
    steering_angle_right = steering_angle - steering_angle_correction

    # add original and flipped images to the list of images
    images.append(image_center)
    images.append(cv2.flip(image_center,1))
    images.append(image_left)
    images.append(cv2.flip(image_left,1))
    images.append(image_right)
    images.append(cv2.flip(image_right,1))

    # add corresponting measurements
    measurements.append(steering_angle)
    measurements.append(-steering_angle)
    measurements.append(steering_angle_left)
    measurements.append(-steering_angle_left)
    measurements.append(steering_angle_right)
    measurements.append(-steering_angle_right)



### Create generator 
def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                make_dataset(batch_sample, images, angles)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
## Creating training and validation generator

train_generator = generator(training_data, batch_size=64)
validation_generator = generator(validation_data, batch_size=64)


## Define architecture

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Conv2D(24, (5, 5), strides =(2,2), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
          
model.add(Conv2D(36, (5, 5), strides =(2,2), activation="relu"))
model.add(Dropout(0.3))
          
model.add(Conv2D(48, (5, 5), strides =(2,2), activation="relu"))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50)) 
model.add(Dense(10))
model.add(Dense(1))

model_image = './images/architecture.png'
plot_model(model, to_file=model_image, show_shapes=True)

## Checkpointer for saving the best model
checkpointer = ModelCheckpoint(filepath='model.h5', 
                               verbose=1, save_best_only=True)

## Compile the model with mse loss and adam optimizer          
model.compile(loss='mse', optimizer='adam')

## Training the model
model.fit_generator(train_generator, validation_steps=len(validation_data), epochs=10, 
                    validation_data=validation_generator, steps_per_epoch= len(training_data),callbacks=[checkpointer])

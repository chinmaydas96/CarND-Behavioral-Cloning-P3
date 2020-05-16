## Import the necessary libraries

import os
import csv
import cv2
import sklearn
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPool2D, Cropping2D, Flatten, Dense,Dropout 
from keras.callbacks import ModelCheckpoint 

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


## Read the csv file for extraction of path of the image

lines = []  
with open('./new_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader :
        lines.append(line)

## Spliting dataset to trainset and validation set
training_data, validation_data = train_test_split(lines,test_size=0.2)

### Create generator 
def generator(lines, batch_size=64):
    len_data = len(lines)
    while 1: 
        shuffle(lines)
        for offset in range(0, len_data, batch_size):
            batch_data = lines[offset:offset+batch_size]
            images = []
            angles = []
            for line in batch_data:
                path = './new_data/IMG/'+line[0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) 
                center_angle = float(line[3])
                images.append(center_image)
                angles.append(center_angle)            
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            
## Creating training and validation generator

train_generator = generator(training_data, batch_size=64)
validation_generator = generator(validation_data, batch_size=64)


## Define architecture

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

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

## Checkpointer for saving the best model
checkpointer = ModelCheckpoint(filepath='model.h5', 
                               verbose=1, save_best_only=True)

## Compile the model with mse loss and adam optimizer          
model.compile(loss='mse', optimizer='adam')

## Training the model
model.fit_generator(train_generator, validation_steps=len(validation_data), epochs=10, 
                    validation_data=validation_generator, steps_per_epoch= len(training_data),callbacks=[checkpointer])

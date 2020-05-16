import os
import csv
import cv2
import sklearn
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Cropping2D
from keras.layers import Flatten, Dense 

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split






os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#this array is used for the atoring of the image names/paths in the csv files
lines = []  
with open('./new_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader :
        lines.append(line)

#using the 10% of data for the validation purpose
training_data, validation_data = train_test_split(lines,test_size=0.2)


def generator(lines, batch_size=32):
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
            
train_generator = generator(training_data, batch_size=32)
validation_generator = generator(validation_data, batch_size=32)

#Using the NVIDIA Model architecture with its specific layer to train the model
#first layer is used for the normalizing of the images
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#the images are cropped in this section in such a way that the trees and the sky part is removed 
#so that there is no invariability present in the images
model.add(Cropping2D(cropping=((70,25),(0,0))))
#A set of convolution layers are there which are used for getting the best possible features
model.add(Conv2D(24, (5, 5), strides =(2,2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides =(2,2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides =(2,2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
#Now the convolutions are done so now the flatten and the repective fully connected layers are present
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50)) 
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, validation_steps=len(validation_data), epochs=8, 
                    validation_data=validation_generator, steps_per_epoch= len(training_data))

model.save('model.h5')
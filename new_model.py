import math
import random
import csv
import sklearn
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, MaxPool2D, Conv2D,  Lambda, Cropping2D
from sklearn.model_selection import train_test_split

# load data from csv file
samples = []
with open("./new_data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


def addDataFromOneLine(line, images, measurements):
    """
    This function reads data from line and adds it to the list of input images
    and measurements
    """

    # correction for left and right camera image
    steering_angle_correction = 0.2

    # load images and measurements
    
    image_center_path = line[0]
    filename_center = image_center_path.split('/')[-1]
    current_path_center = './new_data/IMG/' + filename_center
    
    image_left_path = line[1]
    filename_left = image_left_path.split('/')[-1]
    current_path_center = './new_data/IMG/' + filename_left
    
    image_right_path = line[2]
    filename_right = image_right_path.split('/')[-1]
    current_path_right = './new_data/IMG/' + filename_right
    
    
    
    image_center = np.asarray(Image.open(current_path_center))
    image_left   = np.asarray(Image.open(current_path_center))
    image_right  = np.asarray(Image.open(current_path_right))
    steering_angle = float(line[3])

    # correct angles for left and right image
    steering_angle_left = steering_angle + steering_angle_correction
    steering_angle_right = steering_angle - steering_angle_correction

    # add original and flipped images to the list of images
    images.append(image_center)
    images.append(np.fliplr(image_center))
    images.append(image_left)
    images.append(np.fliplr(image_left))
    images.append(image_right)
    images.append(np.fliplr(image_right))

    # add corresponting measurements
    measurements.append(steering_angle)
    measurements.append(-steering_angle)
    measurements.append(steering_angle_left)
    measurements.append(-steering_angle_left)
    measurements.append(steering_angle_right)
    measurements.append(-steering_angle_right)


def generator(samples, batch_size):
    """
    This function generates one batch of data
    """

    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                addDataFromOneLine(batch_sample, images, angles)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


batch_size = 32

# splitting the data to account for overfitting
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# generators for training and validation
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# model architecture
model = Sequential()
# data preprocessing
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# conv layer 1
model.add(Conv2D(8, (5, 5)))
model.add(MaxPool2D((2, 2)))
model.add(Activation("relu"))
# conv layer 2
model.add(Conv2D(8, (5, 5)))
model.add(MaxPool2D(2, 2))
model.add(Activation("relu"))
# fully connected layer 1
model.add(Flatten())
model.add(Dense(120))
# fully connected layer 2
model.add(Dense(84))
# fully connected layer 3
model.add(Dense(1))

# configure the model
model.compile(loss="mse", optimizer="adam")

# training
model.fit_generator(
    train_generator,
    steps_per_epoch=math.ceil(len(train_samples) / batch_size),
    validation_data=validation_generator,
    validation_steps=math.ceil(len(validation_samples) / batch_size),
    epochs=10,
    verbose=1,
)

# save trained model
model.save("model.h5")
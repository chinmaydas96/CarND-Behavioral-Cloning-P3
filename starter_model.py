import csv
import cv2
import numpy as np


lines = []

with open('image_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []

measurements = []

for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '/Users/bat/image_data/IMG/' + filename
	image =  cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)


X_train = np.array(images)
y_train = np.array(measurements)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Lambda


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train,y_train, validation_split = 0.2, shuffle = True,epochs=5)
model.save('model.h5')
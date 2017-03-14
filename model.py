# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import sklearn
import pickle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#import cv2

# There is 1 output class
nb_classes = 1 


pickle_train = 'train.pickle'
pickle_validation = 'validation.pickle'

batch_size = 32



# load the training dataset from the train pickle 
with open(pickle_train, 'rb') as f:
    #train, test1, _ = ds.getDigitStruct()
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    #delete save to free up memory
    del save
    print('Training set: ', np.array(train_dataset).shape, np.array(train_labels).shape)
    
   
 
# load the validation dataset from the test pickle   
with open(pickle_validation, 'rb') as f:
    save = pickle.load(f)
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    del save
    print('Validation set: ', np.array(valid_dataset).shape, np.array(valid_labels).shape)


def generator(images, angles, batch_size=32):#2):
    num_samples = len(images)
    while True: # Loop forever so the generator never terminates
        images, angles = shuffle(images, angles)
        for offset in range(0, num_samples, batch_size):
            generated_images = images[offset:offset+batch_size]
            generated_angles = angles[offset:offset+batch_size]
            # trim image to only see section with road
            #X_train = np.array(images)
            #y_train = np.array(angles)
            X_train = np.array(generated_images)
            y_train = np.array(generated_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_dataset, train_labels, batch_size=32)
validation_generator = generator(valid_dataset, valid_labels, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

shape = (ch, row, col)

# number of convolutional filters to use
nb_filters = [16, 8, 4, 2]
	

# size of pooling area for max pooling
pool_size = 2

# convolution kernel size
kernel_size = 3

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5 - 1.))

# Starting with the convolutional layer
model.add(Convolution2D(nb_filters[0], kernel_size, kernel_size))
# ReLU Activation
model.add(Activation('elu'))
# The second conv layer will convert 16 channels into 8 channels
model.add(Convolution2D(nb_filters[1], kernel_size, kernel_size))
# ReLU Activation
model.add(Activation('elu'))
# The second conv layer will convert 8 channels into 4 channels
model.add(Convolution2D(nb_filters[2], kernel_size, kernel_size))
# ReLU Activation
model.add(Activation('elu'))
# The second conv layer will convert 4 channels into 2 channels
model.add(Convolution2D(nb_filters[3], kernel_size, kernel_size))
# ReLU Activation
model.add(Activation('elu'))
# Apply Max Pooling for each 2 x 2 pixels
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=None, border_mode='valid', dim_ordering='default'))# MaxPooling2D(pool_size=pool_size))
# Dropout with keep probability 0.5
model.add(Dropout(0.5))

# Flatten the matrix. The input has size of 360
model.add(Flatten())
# Input 360 Output 16
model.add(Dense(512))
# ReLU Activation
model.add(Activation('elu'))
# Dropout with keep probability 0.5
model.add(Dropout(0.5))
# Input 16 Output 16
model.add(Dense(64))
# ReLU Activation
model.add(Activation('elu'))
# Input 16 Output 16
model.add(Dense(16))
# ReLU Activation
model.add(Activation('elu'))
# Dropout with keep probability 0.5
model.add(Dropout(0.5))
# Input 16 Output 1
model.add(Dense(nb_classes))

## Print out summary of the model
model.summary()


model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_dataset), validation_data=validation_generator, nb_val_samples=len(valid_dataset), nb_epoch=6, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#from keras.utils.visualize_util import plot

#plot(model, to_file='model.png', show_shapes=True)

#img = cv2.imread('model.png')

# original image
#plt.subplots(figsize=(5,10))
#plt.subplot(111)
#plt.axis('off')
#plt.imshow(img)

model.save('./model.h5')
print("Model Saved")
    
    
    

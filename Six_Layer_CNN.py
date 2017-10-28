# Source code referenced from : https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
# Adapted to suit the given requirement

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # Smart bar while importing the data
import tflearn # importing tflearn for API Calls
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

IMG_SIZE = 200 # Sam size as defined in 
LR = 1e-3 # Set the learning rate to 0.001
MODEL_NAME = 'vehicle-{}-{}.model'.format(LR, '6conv-basic') # Saving the model for reuse

#Load train data from image processing
train_data = np.load('train_data.npy')

# Real-time data preprocessing, however not considered, due to memory overruns
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
#img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Defining the input data for convolution
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1],data_augmentation=img_aug,  name='input')

# Defining the convolution with ReLu activation and max pooling
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# Defining the fully connected layer with the output of conoluvtion  with ReLu activation and max pooling
convnet = fully_connected(convnet, 100, activation='relu')
#Defining the drop out rate as 0.8 to avoid overfitting and make network more robust
convnet = dropout(convnet, 0.8)

# Output layer configuration with "softmax" activation
convnet = fully_connected(convnet, 3, activation='softmax')

# Defining the back propagation parameters with learning rate 0.001, gradient descent optimisation using "adaptive momentum"
# Computing the loss using cross entropy
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log', verbose = '3')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

# splitting the training and validation data, from train data:

train = train_data[:-300]
test = train_data[-300:]

# loading the dataset with labels
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
#print X --> Debug
Y = [i[1] for i in train]
#print Y --> Debug

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
#print test_x --> Debug
test_y = [i[1] for i in test]
#print test_y --> Debug

# Now we fit for 10 epochs :

model.fit({'input': X}, {'targets': Y}, n_epoch=10, batch_size=10, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=300, show_metric=True, run_id=MODEL_NAME)

#Save the model
model.save(MODEL_NAME)

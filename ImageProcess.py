# Source : Image processing approach is derived based on the kats-vs-dogs machine learning tutorial
# https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/

import cv2                 # Opencv-python to work with images
import numpy as np         # dealing with arrays and to store the data in arrays
import os                  # Support directory paths
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # smart percentage bar for tasks. 

TRAIN_DIR = '/home/ec2-user/image_data/Train'
TEST_DIR = '/home/ec2-user/image_data/Valid'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'vehicle-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match

def label_img(img):
   # print img
   # label name is being sourced from first three letters of image name
    word_label = img.split('_')[-2]
    # print("Word label: ", word_label) --> for debug
# conversion to one-hot array [Car,Truck, Bike]
   
    if word_label == 'Bik': return [0,0,1]
    elif word_label == 'car': return [0,1,0]
    elif word_label == 'Tru': return [1,0,0]
	

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        # print("Label ", label) --> for debugging
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
	
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
	# Ensuring that image number is considered
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    np.save('test_data.npy', testing_data)
    return testing_data
	
train_data = create_train_data()
print ("Training and validation data is created")
test_data = process_test_data()
print ("Testing data is created")


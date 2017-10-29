# importing matplot library for plotting function 
import matplotlib
matplotlib.use('Agg') # Support non interactive backends

from pandas_ml import ConfusionMatrix # Confusion matrix 
import matplotlib.pyplot as plt
import numpy as np
import os # to read the directories, especially to load the model 
import tensorflow as tf
import tflearn
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

IMG_SIZE = 200
TEST_DIR = '/home/ubuntu/src/datta_ms/Test'

# Loading the test data generated from image process.py
test_data = np.load('test_data.npy')

LR = 1e-3
MODEL_NAME = 'vehicle-{}-{}.model'.format(LR, '6conv-basic') 

# labelling the image
def label_img(img):
   # print img
    word_label = img.split('_')[-2]
   # print("Word label: ", word_label)   
   # conversion to one-hot array [Car,Truck, Bike]
   
    if word_label == 'Bik': return 'Bike'
    elif word_label == 'car': return 'Car'
    elif word_label == 'Tru': return 'Truck'

# Defining the network 
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

# Load the trained model 
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

# Labelling for the entire test data, i.e true data

def testlbltrue_data():
    true_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
       #print("True Label", label)
       # exit(0)
        path = os.path.join(TEST_DIR,img)
        true_data.append([np.array(label)])
    return true_data

def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum() 	

# Predicted data for test dataset from the model 

def testlblpred_data():
    pred_data = []
    for num,data in enumerate(test_data[:677]):
        img_data = data[0]
        #print img_data
        #exit(0)
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        #print model_out
       
        output = softmax(model_out)
        #print output
        if np.argmax(model_out) == 1: str_label='Car'
        elif np.argmax(model_out) == 0: str_label='Truck'
        else: str_label='Bike'
        pred_data.append([np.array(str_label)])
        #print ("Pred label: ", str_label)
    return pred_data

y_true = testlbltrue_data()
print y_true

# Result is pushed to files 
with open('true.txt','wb') as f:
    np.savetxt(f,y_true,fmt='%s',delimiter=',')

y_pred = testlblpred_data()
print y_pred

#exit (0)
with open('pred.txt','wb') as f:
    np.savetxt(f,y_pred,fmt='%s',delimiter=',')

# Confusion matrix 
confusion_matrix = ConfusionMatrix(y_true, y_pred)
print("Confusion matrix:\n%s" % confusion_matrix)

# Plotting confusion matrix
confusion_matrix.plot()

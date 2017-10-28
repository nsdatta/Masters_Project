# 
# Building an algorithm based on back propagation usint Tensorflow : Sigmoid

import tensorflow
import tensorflow as tf
import numpy as np
import sys

IMG_SIZE = 200

# Loading the training data

train_data = np.load('train_data.npy')

a_0 = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE])
print a_0
y = tf.placeholder(tf.float32, [None, 3])
print y

middle = 50

w_1 = tf.Variable(tf.truncated_normal([IMG_SIZE*IMG_SIZE, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, middle]))
b_2 = tf.Variable(tf.truncated_normal([1, middle]))
w_3 = tf.Variable(tf.truncated_normal([middle, 3]))
b_3 = tf.Variable(tf.truncated_normal([1, 3]))

def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))

z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)

a_2 = sigma(z_2)
z_3 = tf.add(tf.matmul(a_2, w_3), b_3)
a_3 = sigma(z_3)

diff = tf.subtract(a_3, y)

def sigmaprime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))

d_z_3 = tf.multiply(diff, sigmaprime(z_3))
d_b_3 = d_z_3
d_w_3 = tf.matmul(tf.transpose(a_2), d_z_3)

d_a_2 = tf.matmul(d_z_3, tf.transpose(w_3))
d_z_2 = tf.multiply(d_a_2, sigmaprime(z_2))
d_b_2 = d_z_2
d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
d_z_1 = tf.multiply(d_a_1, sigmaprime(z_1))
d_b_1 = d_z_1
d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)

eta = tf.constant(0.005)
step = [
    tf.assign(w_1,
            tf.subtract(w_1, tf.multiply(eta, d_w_1)))
  , tf.assign(b_1,
            tf.subtract(b_1, tf.multiply(eta,
                               tf.reduce_mean(d_b_1, axis=[0]))))
  , tf.assign(w_2,
            tf.subtract(w_2, tf.multiply(eta, d_w_2)))
  , tf.assign(b_2,
            tf.subtract(b_2, tf.multiply(eta,
                               tf.reduce_mean(d_b_2, axis=[0]))))
   , tf.assign(w_3,
            tf.subtract(w_3, tf.multiply(eta, d_w_3)))
  , tf.assign(b_3,
            tf.subtract(b_3, tf.multiply(eta,
                               tf.reduce_mean(d_b_3, axis=[0]))))
]

acct_mat = tf.equal(tf.argmax(a_3, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train = train_data[:-300]
test = train_data[-300:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE*IMG_SIZE)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE*IMG_SIZE)
test_y = [i[1] for i in test]

f = open("sig_backprop_result.txt", 'w')
sys.stdout = f

for i in xrange(2700):
    
    x_p= np.reshape(X[i],(1,IMG_SIZE*IMG_SIZE))
    y_p= np.reshape(Y[i],(1,3))
    sess.run(step, feed_dict = {a_0: x_p ,
                                y : y_p})
    if i % 100 == 0:
        res = sess.run(acct_res, feed_dict =
                       {a_0: test_x[:100],
                        y :  test_y[:100]})
	print res

f.close()


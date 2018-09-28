import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py

def normalize_with_moments(x, axes=[0, 1], epsilon=1e-8):
    mean, variance = tf.nn.moments(x, axes=axes)
    x_normed = (x - mean) / tf.sqrt(variance + epsilon) # epsilon to avoid dividing by zero
    return x_normed

def correlation(X,Y):
    norm_X = normalize_with_moments(X, axes=[0])
    norm_Y = normalize_with_moments(Y, axes=[0])
    n = tf.shape(norm_X)[0]
    return tf.matmul(tf.transpose(norm_X),norm_Y)/tf.cast(n,dtype=tf.float32)

def GraphNet_regularization(W, X, lambda1=0.001):
    corr_X = correlation(X,X)
    diag_X = tf.diag(tf.diag_part(corr_X))
    X1 = corr_X-diag_X
    L1 = tf.diag(tf.reduce_sum(X1,0))
    H1 = X1-L1
    P = lambda1*tf.norm(W,ord=1) + lambda1*tf.matmul(tf.matmul(tf.transpose(W),H1),W)
    return P


# Load datasets
X1_data=h5py.File('./data/smc.mat')
X = np.transpose(np.mat(X1_data['X']))
Y = np.transpose(np.mat(X1_data['Y']))

# view1 layers
X1 = tf.placeholder(tf.float32, [None,27120])
X2 = tf.placeholder(tf.float32, [None,1])

W1 = tf.Variable(tf.random_uniform([27120,1],-1,1))
L1 = tf.matmul(X1,W1)

cost = tf.reduce_mean(tf.squared_difference(L1, X2)) + 0.1*tf.norm(W1,ord=1) + 0.01*GraphNet_regularization(W1,X1)
#-1*correlation(X2,L4) + 0.1*tf.norm(W1,ord=1)


# basic settings
batch_size =28
n_epochs = 100

optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
err1 = 1
for epoch in range(n_epochs):

    _, err2 = sess.run([optimizer, cost], feed_dict={X1:X,X2:Y})
    obj  = err1 - err2
    err1 = err2

    print("iteration:", epoch, "loss:", err, "obj", obj ,end='\r')



W = sess.run(W1,feed_dict={X1: X ,X2: Y})


import scipy.io
scipy.io.savemat('xx.mat', {'output': W})

plt.figure()
x = np.linspace(0,len(W),len(W))
markerline, stemlines, baseline = plt.stem(x, W, '-.')

plt.figure()
x = np.linspace(0,1024,1024)
markerline, stemlines, baseline = plt.stem(x, X2proj, '-.')

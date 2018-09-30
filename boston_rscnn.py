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


# We use a simple regression dataset taken from scikit-learn
from sklearn import datasets
data = datasets.load_boston()

# Preprocess the inputs to be in [-1,1] and split the data in train/test sets
from sklearn import preprocessing, model_selection
X = preprocessing.MinMaxScaler(feature_range=(-1,+1)).fit_transform(data['data'])
y = preprocessing.MinMaxScaler().fit_transform(data['target'].reshape(-1, 1))
train_X1, test_X1, train_X2, test_X2 = model_selection.train_test_split(X, y, test_size=0.25)

# view1 layers
X1 = tf.placeholder(tf.float32, [None,13,1])
X2 = tf.placeholder(tf.float32, [None,1])

W1 = tf.Variable(tf.random_normal([3, 1, 16], stddev = 0.01))
L1 = tf.nn.conv1d(tf.nn.relu(X1),W1,1,padding='SAME')

L2 = tf.contrib.layers.flatten(L1)
W2 = tf.Variable(tf.random_uniform([16*13,1],-0.5,0.5))
L3 = tf.matmul(L2,W2)

cost = -1*correlation(X2,L3) +  0.01*tf.norm(W1,ord=1) + 0.01*tf.norm(W2,ord=1)


# basic settings
batch_size = len(train_X1)
n_epochs = 100

optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

err1 = 100
iterations = 0
for epoch in range(n_epochs):
    index = np.arange(train_X1.shape[0])
    np.random.shuffle(index)
    trX1 = train_X1[index]
    trX2 = train_X2[index]

    # train
    corr_train = 0
    for current_batch_index in range(0,len(train_X1),batch_size):
        current_batch_X1 = trX1[current_batch_index:current_batch_index+batch_size,:]
        current_batch_X2 = trX2[current_batch_index:current_batch_index+batch_size,:]
        current_batch_X1 = current_batch_X1.reshape(-1,13,1)
        _, neg_corr_val = sess.run([optimizer, cost], feed_dict={X1:current_batch_X1,X2:current_batch_X2})

        corr_train = corr_train + neg_corr_val
        iterations += 1


        #tune_neg_corr_val = sess.run(cost,feed_dict={X1: test_X1[0:1000,:],X2: test_X2[0:1000,:]})"neg_loss_for_tune:", tune_neg_corr_val,
    err2 = corr_train / (len(train_X1)/batch_size)
    err = (err1-err2)/err1
    err1 = err2
    test_X1 = test_X1.reshape(-1,13,1)
    corr_val = sess.run(cost,feed_dict={X1: test_X1,X2: test_X2})
    print("iteration:", epoch, "% err:", abs(err), "validation_loss:", corr_val)
    if abs(err) < 0.01:
        print("iteration:", epoch, "% err:", abs(err), "validation_loss:", corr_val)
        break

train_X1 = train_X1.reshape(-1,13,1)
W1 = sess.run(W1,feed_dict={X1: train_X1 ,X2: train_X2})
W2 = sess.run(W2,feed_dict={X1: train_X1 ,X2: train_X2})


import scipy.io
scipy.io.savemat('W1.mat', {'W1': W1})
scipy.io.savemat('W2.mat', {'W2': W2})
scipy.io.savemat('X.mat', {'X': train_X1})
scipy.io.savemat('Y.mat', {'Y': train_X2})

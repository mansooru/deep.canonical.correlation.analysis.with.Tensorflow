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
X1 = tf.placeholder(tf.float32, [None,13])
X2 = tf.placeholder(tf.float32, [None,1])

W1 = tf.Variable(tf.random_uniform([13, 32],-1,1))
L1 = tf.nn.relu(tf.matmul(X1,W1))

W2 = tf.Variable(tf.random_uniform([32, 32],-1,1))
L2 = tf.nn.relu(tf.matmul(L1,W2))

W3 = tf.Variable(tf.random_uniform([32,13],-1,1))
L3 = tf.nn.relu(tf.matmul(L2,W3))

W4 = tf.Variable(tf.random_uniform([13,1],-1,1))
L4 = tf.matmul(L3,W4)

cost = tf.reduce_mean(tf.squared_difference(L4, X2)) + 0.1*tf.norm(W1,ord=1)
#-1*correlation(X2,L4) + 0.1*tf.norm(W1,ord=1)


# basic settings
batch_size = len(train_X1)
n_epochs = 1000

optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

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
        _, neg_corr_val = sess.run([optimizer, cost], feed_dict={X1:current_batch_X1,X2:current_batch_X2})

        corr_train = corr_train + neg_corr_val
        iterations += 1


        #tune_neg_corr_val = sess.run(cost,feed_dict={X1: test_X1[0:1000,:],X2: test_X2[0:1000,:]})"neg_loss_for_tune:", tune_neg_corr_val,
    corr_train = corr_train / (len(train_X1)/batch_size)
    corr_val = sess.run(cost,feed_dict={X1: test_X1,X2: test_X2})
    print("iteration:", epoch, "train_loss:", corr_train, "validation_loss:", corr_val, end='\r')



W = sess.run(W1,feed_dict={X1: train_X1 ,X2: train_X2})
W

plt.figure()
x = np.linspace(0,len(W),len(W))
markerline, stemlines, baseline = plt.stem(x, W, '-.')

plt.figure()
x = np.linspace(0,1024,1024)
markerline, stemlines, baseline = plt.stem(x, X2proj, '-.')

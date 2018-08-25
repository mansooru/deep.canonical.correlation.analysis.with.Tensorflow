import tensorflow as tf
import numpy as np
from neg_corr import neg_correlation
import h5py
import matplotlib.pyplot as plt


# basic settings
n_epochs = 100
learning_rate = 0.01
batch_size = 100

# load DataSet
X1_data=h5py.File('./data/train_X1_dataset.ver1.mat')
train_X1 = np.mat(X1_data['input'])
train_X1_label = np.transpose(np.mat(X1_data['label']))


X2_data=h5py.File('./data/train_X2_dataset.ver1.mat')
train_X2 = np.mat(X2_data['input'])
train_X2_label = np.transpose(np.mat(X2_data['label']))

# view1 layers
X1 = tf.placeholder(tf.float32, [None,1024])
X2 = tf.placeholder(tf.float32, [None,1024])

W11 = tf.Variable(tf.random_uniform([1024, 2048],-1,1))
L11 = tf.nn.relu(tf.matmul(X1,W11))

W21 = tf.Variable(tf.random_uniform([2048, 2048],-1,1))
L21 = tf.nn.relu(tf.matmul(L11,W21))

W31 = tf.Variable(tf.random_uniform([2048, 1024],-1,1))
L31 = tf.nn.relu(tf.matmul(L21,W31))

W41 = tf.Variable(tf.random_uniform([1024, 1],-1,1))
model1 = tf.matmul(L31,W41)

# view2 layers
W12 = tf.Variable(tf.random_uniform([1024, 2048],-1,1))
L12 = tf.nn.relu(tf.matmul(X2,W12))

W22 = tf.Variable(tf.random_uniform([2048, 2048],-1,1))
L22 = tf.nn.relu(tf.matmul(L12,W22))

W32 = tf.Variable(tf.random_uniform([2048, 1024],-1,1))
L32 = tf.nn.relu(tf.matmul(L22,W32))

W42 = tf.Variable(tf.random_uniform([1024, 1],-1,1))
model2 = tf.matmul(L32,W42)


cost = tf.reduce_mean(tf.square(W41 - train_X1_label)) + tf.reduce_mean(tf.square(W42 - train_X2_label)) + neg_correlation(model1,model2,1)
# neg_correlation(model1,model2,1) ++ 0.01*tf.norm(W41,ord=1) + 0.01*tf.norm(W42,ord=1)
#+  0.01*tf.norm(W11,ord=2) + 0.1*tf.norm(W12,ord=2)+ 0.01*tf.norm(W21,ord=1) + 0.01*tf.norm(W22,ord=1) + 0.01 * tf.norm(W31,ord=1) + 0.01*tf.norm(W32,ord=1)
#cost = tf.contrib.metrics.streaming_pearson_correlation(model1,model2)
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
    for current_batch_index in range(0,len(train_X1),batch_size):
        current_batch_X1 = trX1[current_batch_index:current_batch_index+batch_size,:]
        current_batch_X2 = trX2[current_batch_index:current_batch_index+batch_size,:]
        _, neg_corr_val = sess.run([optimizer, cost], feed_dict={X1:current_batch_X1,X2:current_batch_X2})


        if iterations % 1 == 0:
            print("iteration:", iterations, "neg_loss_for_train:", neg_corr_val, end='\r')
            #tune_neg_corr_val = sess.run(cost,feed_dict={X1: tune_X1,X2: tune_X2})
            #print("neg_loss_for_tune:", tune_neg_corr_val)

        iterations += 1

X1proj, X2proj = sess.run([W41,W42],feed_dict={X1: train_X1 ,X2: train_X2})


plt.figure()
x = np.linspace(0,1024,1024)
markerline, stemlines, baseline = plt.stem(x, X1proj, '-.')


X1proj, X2proj = sess.run([W41,W42],feed_dict={X1: train_X1 ,X2: train_X2})


plt.figure()
x = np.linspace(0,1024,1024)
markerline, stemlines, baseline = plt.stem(x, X2proj, '-.')

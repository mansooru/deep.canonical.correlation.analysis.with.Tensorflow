import tensorflow as tf
import scipy.io as sio
from linear_cca import linear_cca
import numpy as np
from neg_corr import neg_correlation
from sklearn import svm
from sklearn.metrics import accuracy_score

# basic settings
n_epochs = 100
learning_rate = 0.01
momentum=0.99
batch_size = 800

# load DataSet
data=sio.loadmat('MNIST.mat')
train_X1 = data['X1']
train_X2 = data['X2']
trainLabel = data['trainLabel']

tune_X1 = data['XV1']
tune_X2 = data['XV2']
tuneLabel = data['tuneLabel']

test_X1 = data['XTe1']
test_X2 = data['XTe2']
testLabel = data['testLabel']

# view1 layers
h11 = tf.layers.dense(X1, 784,activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
h21 = tf.layers.dense(h11, 1024,activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
h31 = tf.layers.dense(h21, 1024,activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
h41 = tf.layers.dense(h31, 10, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

# view2 layers
h12 = tf.layers.dense(X2, 784,activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
h22 = tf.layers.dense(h12, 1024,activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
h32 = tf.layers.dense(h22, 1024,activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
h42 = tf.layers.dense(h32, 10, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

cost = neg_correlation(h41,h42,10)
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

iterations = 0
for epoch in range(n_epochs):

    for start, end in zip(range(0, train_X1.shape[0], batch_size), range(batch_size, train_X1.shape[0], batch_size)):
        Xs1 = train_X1[start:end]
        Xs2 = train_X2[start:end]

        _, neg_corr_val = sess.run([optimizer, cost], feed_dict={X1:Xs1,X2:Xs2})


        if iterations % 100 == 0:
            print("iteration:", iterations)
            print("neg_loss_for_train:", neg_corr_val)
            tune_neg_corr_val = sess.run(cost,feed_dict={X1: tune_X1,X2: tune_X2})
            print("neg_loss_for_tune:", tune_neg_corr_val)

        iterations += 1

X1proj, X2proj = sess.run([h41,h42],feed_dict={X1: train_X1 ,X2: train_X2})
XV1proj, XV2proj = sess.run([h41,h42],feed_dict={X1: tune_X1,X2: tune_X2})
XTe1proj, XTe2proj = sess.run([h41,h42],feed_dict={X1: test_X1,X2: test_X2})

print("Linear CCA started!")
w = [None, None]
m = [None, None]
w[0], w[1], m[0], m[1] = linear_cca(X1proj, X2proj, 10)
print("Linear CCA ended!")
X1proj -= m[0].reshape([1, -1]).repeat(len(X1proj), axis=0)
X1proj = np.dot(X1proj, w[0])

XV1proj -= m[0].reshape([1, -1]).repeat(len(XV1proj), axis=0)
XV1proj = np.dot(XV1proj, w[0])

XTe1proj -= m[0].reshape([1, -1]).repeat(len(XTe1proj), axis=0)
XTe1proj = np.dot(XTe1proj, w[0])

################# SVM classify #############################

print('training SVM...')
clf = svm.LinearSVC(C=0.01, dual=False)
clf.fit(X1proj, trainLabel.ravel())

p = clf.predict(XTe1proj)
test_acc = accuracy_score(testLabel, p)
p = clf.predict(XV1proj)
valid_acc = accuracy_score(tuneLabel, p)
print('DCCA: tune acc={}, test acc={}'.format(valid_acc, test_acc))

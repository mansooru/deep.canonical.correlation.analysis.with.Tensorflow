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
X1 = tf.placeholder(tf.float32, [None,784])
X2 = tf.placeholder(tf.float32, [None,784])

W11 = tf.Variable(tf.random_uniform([784, 1024],-1,1))
B11 = tf.Variable(tf.random_uniform([784, 1024],-1,1))
L11 = tf.nn.relu(tf.add(tf.matmul(X1,W11),B11))

W21 = tf.Variable(tf.random_uniform([1024, 1024],-1,1))
B21 = tf.Variable(tf.random_uniform([1024, 1024],-1,1))
L21 = tf.nn.relu(tf.add(tf.matmul(L11,W21),B21))

W31 = tf.Variable(tf.random_uniform([1024, 1024],-1,1))
B31 = tf.Variable(tf.random_uniform([1024, 1024],-1,1))
L31 = tf.nn.relu(tf.add(tf.matmul(L21,W31),B31))

W41 = tf.Variable(tf.random_uniform([1024, 10],-1,1))
B41 = tf.Variable(tf.random_uniform([1024, 10],-1,1))
model1 = tf.add(tf.matmul(L31,W41),B41)

# view2 layers
W12 = tf.Variable(tf.random_uniform([784, 1024],-1,1))
B12 = tf.Variable(tf.random_uniform([784, 1024],-1,1))
L12 = tf.nn.relu(tf.add(tf.matmul(X2,W12),B12))

W22 = tf.Variable(tf.random_uniform([1024, 1024],-1,1))
B22 = tf.Variable(tf.random_uniform([1024, 1024],-1,1))
L22 = tf.nn.relu(tf.add(tf.matmul(L12,W22),B22))

W32 = tf.Variable(tf.random_uniform([1024, 1024],-1,1))
B32 = tf.Variable(tf.random_uniform([1024, 1024],-1,1))
L32 = tf.nn.relu(tf.add(tf.matmul(L22,W32),B32))

W42 = tf.Variable(tf.random_uniform([1024, 10],-1,1))
B42 = tf.Variable(tf.random_uniform([1024, 10],-1,1)
model2 = tf.add(tf.matmul(L32,W42),B42)

cost = neg_correlation(model1,model2,10)
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

X1proj, X2proj = sess.run([model1, model2],feed_dict={X1: train_X1 ,X2: train_X2})
XV1proj, XV2proj = sess.run([model1, model2],feed_dict={X1: tune_X1,X2: tune_X2})
XTe1proj, XTe2proj = sess.run([model1, model2],feed_dict={X1: test_X1,X2: test_X2})

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

trainLable = trainData.labels.astype('float')
tuneLable = tuneData.labels.astype('float')
testLable = testData.labels.astype('float')


################# SVM classify #############################

print('training SVM...')
clf = svm.LinearSVC(C=0.01, dual=False)
clf.fit(X1proj, trainLabel.ravel())

p = clf.predict(XTe1proj)
test_acc = accuracy_score(testLabel, p)
p = clf.predict(XV1proj)
valid_acc = accuracy_score(tuneLabel, p)
print('DCCA: tune acc={}, test acc={}'.format(valid_acc, test_acc))

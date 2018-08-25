import tensorflow as tf
import numpy as np
from neg_corr import neg_correlation
import matplotlib.pyplot as plt
import h5py

# basic settings
n_epochs = 70
learning_rate = 0.001
batch_size = 200

# load DataSet
X1_data=h5py.File('./data/train_X1_dataset.ver5.mat')
train_X1 = np.transpose(np.mat(X1_data['input_X1']))
train_X1_label = np.transpose(np.mat(X1_data['label_X1']))
H1 = np.mat(X1_data['H1'], dtype='float32')

X2_data=h5py.File('./data/train_X2_dataset.ver5.mat')
train_X2 = np.transpose(np.mat(X2_data['input_X2']))
train_X2_label = np.transpose(np.mat(X2_data['label_X2']))
H2 = np.mat(X2_data['H2'],dtype='float32')

test_X1_data=h5py.File('./data/test_X1_dataset.ver5.mat')
test_X1 = np.transpose(np.mat(test_X1_data['input_X1']))
test_X1_label = np.transpose(np.mat(test_X1_data['label_X1']))


test_X2_data=h5py.File('./data/test_X2_dataset.ver5.mat')
test_X2 = np.transpose(np.mat(test_X2_data['input_X2']))
test_X2_label = np.transpose(np.mat(test_X2_data['label_X2']))


# view1 layers
X1 = tf.placeholder(tf.float32, [None,1024])
X2 = tf.placeholder(tf.float32, [None,1024])

W11 = tf.Variable(tf.random_uniform([1024, 1],-1,1))
#L11 = tf.nn.relu(tf.matmul(X1,W11))
model1 = tf.matmul(X1,W11)

# view2 layers
W12 = tf.Variable(tf.random_uniform([1024, 1],-1,1))
#L12 = tf.nn.relu(tf.matmul(X2,W12))
model2 = tf.matmul(X2,W12)

cost = neg_correlation(model1,model2,1) + 0.1*tf.norm(W11,ord=1) + 0.1*tf.norm(W12,ord=1) + 0.1*tf.matmul(tf.matmul(tf.transpose(W11),H1),W11) + 0.1*tf.matmul(tf.matmul(tf.transpose(W12),H2),W12)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

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

X1proj, X2proj = sess.run([W11,W12],feed_dict={X1: train_X1 ,X2: train_X2})


plt.figure()
x = np.linspace(0,1024,1024)
markerline, stemlines, baseline = plt.stem(x, X1proj, '-.')

plt.figure()
x = np.linspace(0,1024,1024)
markerline, stemlines, baseline = plt.stem(x, X2proj, '-.')

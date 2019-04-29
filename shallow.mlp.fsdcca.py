# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:22:11 2019

@author: mskim
"""

import tensorflow as tf
import numpy as np
from get_graphlaplaplacian import Graph_laplacian
import h5py
import matplotlib.pyplot as plt

# basic settings
n_epochs = 100
learning_rate = 0.01
batch_size = 20

# load DataSet
X1_data=h5py.File('./02.data/train_X_cate_img_v1.mat')
train_X1 = np.transpose(np.mat(X1_data['X']))
X1_test_data=h5py.File('./02.data/train_X_cate_img_v5.mat')
test_X1 = np.transpose(np.mat(X1_test_data['X1']))
X2_data=h5py.File('./02.data/train_Y_cate_img_v1.mat')
train_X2 = np.transpose(np.mat(X2_data['Y']))
X2_test_data=h5py.File('./02.data/train_Y_cate_img_v5.mat')
test_X2 = np.transpose(np.mat(X2_test_data['Y1']))

n_snp_features = 500
n_img_features = 100
n_inter_features = 1500

# Gan for snp to img : Gsi 
def G_a(X):
    with tf.variable_scope("generator_a", reuse=tf.AUTO_REUSE ):
        W11 = tf.get_variable("Ga1", shape=[n_snp_features, n_inter_features], 
                              initializer=tf.contrib.layers.xavier_initializer())
        L11 = tf.nn.relu(tf.matmul(X,W11))
        
        W12 = tf.get_variable("Ga2", shape=[n_inter_features, n_img_features], 
                              initializer=tf.contrib.layers.xavier_initializer())
        L12 = tf.nn.relu(tf.matmul(L11,W12))
         
        W13 = tf.get_variable("Ga3", shape=[1, n_img_features], 
                              initializer=tf.contrib.layers.xavier_initializer())
        W13 = W13/tf.norm(tf.matmul(W13,tf.transpose(X2)),ord=2)
        L13 = tf.multiply(L12,W13)
        
    return W13, L13


# Gan for img to snp : Gis 
def G_b(Y):
    with tf.variable_scope("generator_b", reuse=tf.AUTO_REUSE ):
        W21 = tf.get_variable("Gb1", shape=[n_img_features, n_inter_features], 
                              initializer=tf.contrib.layers.xavier_initializer())
        L21 = tf.nn.relu(tf.matmul(Y,W21))
        
        W22 = tf.get_variable("Gb2", shape=[n_inter_features, n_snp_features], 
                              initializer=tf.contrib.layers.xavier_initializer())
        L22 = tf.nn.relu(tf.matmul(L21,W22))
        
        W23 = tf.get_variable("Gb3", shape=[1, n_snp_features], 
                              initializer=tf.contrib.layers.xavier_initializer())
        W23 = W23/tf.norm(tf.matmul(W23,tf.transpose(X1)),ord=2)
        L23 = tf.multiply(L22,W23)
    
    return W23, L23

# Build GAN model    
X1 = tf.placeholder(tf.float32, [None, n_snp_features])
X2 = tf.placeholder(tf.float32, [None, n_img_features])
PX = Graph_laplacian(tf.matmul(tf.transpose(X1),X1))
PY = Graph_laplacian(tf.matmul(tf.transpose(X2),X2))

_, genA_tmp = G_a(X1)   
Fs_snp, genAB = G_b(genA_tmp)    # snp -> img' -> snp'

_, genB_tmp = G_b(X2) 
Fs_img, genBA = G_a(genB_tmp)    # img -> snp' -> img'

n_epochs = 20
alpha = 0.4
loss = tf.reduce_mean(tf.square(genBA-X2)) + tf.reduce_mean(tf.square(genAB-X1))
#corr,_ = tf.contrib.metrics.streaming_pearson_correlation(tf.matmul(X1,tf.transpose(Fs_snp)),tf.matmul(X2,tf.transpose(Fs_img)))
corr = tf.matmul(tf.matmul(Fs_img,tf.transpose(X2)),tf.matmul(X1,tf.transpose(Fs_snp)))

cost = (alpha)*tf.reduce_mean(tf.square(genBA-X2))- (1-alpha)*corr + 0.8*tf.norm(Fs_img,ord=1) + 0.03*tf.matmul(tf.matmul(Fs_img,PY),tf.transpose(Fs_img))
cost1 = (alpha)*tf.reduce_mean(tf.square(genAB-X1)) - (1-alpha)*corr  + 0.4*tf.norm(Fs_snp,ord=1) + 0.05*tf.matmul(tf.matmul(Fs_snp,PX),tf.transpose(Fs_snp))
#+ 0.01*tf.matmul(tf.matmul(Fs_img,PY),tf.transpose(Fs_img))+ 0.01*tf.matmul(tf.matmul(Fs_snp,PX),tf.transpose(Fs_snp))

with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    #optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    optimizer = tf.train.AdagradOptimizer(0.06).minimize(cost)
    optimizer1 = tf.train.AdagradOptimizer(0.2).minimize(cost1)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

result = np.zeros([50000,3])
iterations = 0
for epoch in range(n_epochs):
    index = np.arange(train_X1.shape[0])
    np.random.shuffle(index)
    trX1 = train_X1[index]
    trX2 = train_X2[index]

    # train
    for current_batch_index in range(0,train_X1.shape[0],batch_size):
        current_batch_X1 = trX1[current_batch_index:current_batch_index+batch_size,:]
        current_batch_X2 = trX2[current_batch_index:current_batch_index+batch_size,:]
        _, train_loss = sess.run([optimizer, cost], feed_dict={X1:current_batch_X1, X2:current_batch_X2})
        _, train_loss1 = sess.run([optimizer1, cost1], feed_dict={X1:current_batch_X1, X2:current_batch_X2})
        cor, test_loss = sess.run([corr, cost], feed_dict={X1:test_X1, X2:test_X2})
        
        result[iterations,0] = train_loss 
        result[iterations,1] = test_loss 
        result[iterations,2] = cor
        
        loss1 = train_loss+train_loss1

        if iterations % 1 == 0:
            print("iteration:", iterations, "train_loss: ", '%.2f' % loss1, "test_loss: ", '%.2f' % test_loss, "test_corr: ", '%.2f' % cor, end='\n')

        iterations += 1
              
        
        
# Results

X1_label = np.mat(X2_data['v1_gt'])
plt.figure()
x = np.linspace(0,100,100)
markerline, stemlines, baseline = plt.stem(x,X1_label, '.-')


X1proj = sess.run(Fs_img,feed_dict={X1: test_X1 ,X2: test_X2})
plt.figure()
x = np.linspace(0,100,100)
markerline, stemlines, baseline = plt.stem(x,np.transpose(X1proj), '.-')



X2_label = np.mat(X1_data['u0_gt'])
plt.figure()
x = np.linspace(0,500,500)
markerline, stemlines, baseline = plt.stem(x,X2_label, '.-')


X2proj = sess.run(Fs_snp,feed_dict={X1: test_X1 ,X2: test_X2})

print('image')
plt.figure()
x = np.linspace(0,500,500)
markerline, stemlines, baseline = plt.stem(x,np.transpose(X2proj), '.-')






















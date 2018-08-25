# U-net based canonical correlation analysis

import tensorflow as tf
import numpy as np,sys,os
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import h5py
from cca import ccaSvd

#np.random.seed(678)
#tf.set_random_seed(1400)

# load DataSet
X1_data=h5py.File('./data/train_X1_dataset.mat')
train_X1 = X1_data['input']
train_X1_label = X1_data['label']
train_X1 = np.transpose(train_X1)
train_X1 = np.expand_dims(train_X1[:,:],axis=2)
train_X1 = np.expand_dims(train_X1[:,:],axis=2)


X2_data=h5py.File('./data/train_X2_dataset.mat')
train_X2 = X2_data['input']
train_X2_label = X2_data['label']
train_X2 = np.transpose(train_X2)
train_X2 = np.expand_dims(train_X2[:,:],axis=2)
train_X2 = np.expand_dims(train_X2[:,:],axis=2)

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(s): return tf.cast(tf.greater(s,0),dtype=tf.float32)
def tf_softmax(x): return tf.nn.softmax(x)
def np_sigmoid(x): 1/(1 + np.exp(-1 *x))

# --- make class ---
class conlayer_left():

    def __init__(self,ker_x,ker_y,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker_x,ker_y,in_c,out_c],stddev=0.05))

    def feedforward(self,input,stride=1,dilate=1):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,1,1],padding='SAME')
        self.layerA = tf_relu(self.layer)
        return self.layerA

class conlayer_right():

    def __init__(self,ker_x,ker_y,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker_x,ker_y,in_c,out_c],stddev=0.05))

    def feedforward(self,input,stride=1,dilate=1,output=1):
        self.input  = input

        current_shape_size = input.shape

        self.layer = tf.nn.conv2d_transpose(input,self.w,output_shape=[batch_size] + [int(current_shape_size[1].value*2),int(current_shape_size[2].value),int(current_shape_size[3].value/2)],strides=[1,2,1,1],padding='SAME')
        self.layerA = tf_relu(self.layer)
        return self.layerA

# --- hyper ---
num_epoch = 20
init_lr = 0.0001
batch_size = 20

# --- make layer ---
# left
l1_1 = conlayer_left(3,1,1,3)
l1_2 = conlayer_left(3,1,3,3)
l1_3 = conlayer_left(3,1,3,3)

l2_1 = conlayer_left(3,1,3,6)
l2_2 = conlayer_left(3,1,6,6)
l2_3 = conlayer_left(3,1,6,6)

l3_1 = conlayer_left(3,1,6,12)
l3_2 = conlayer_left(3,1,12,12)
l3_3 = conlayer_left(3,1,12,12)

l4_1 = conlayer_left(3,1,12,24)
l4_2 = conlayer_left(3,1,24,24)
l4_3 = conlayer_left(3,1,24,24)

l5_1 = conlayer_left(3,1,24,48)
l5_2 = conlayer_left(3,1,48,48)
l5_3 = conlayer_left(3,1,48,24)

# right
l6_1 = conlayer_right(3,1,24,48)
l6_2 = conlayer_left(3,1,24,24)
l6_3 = conlayer_left(3,1,24,12)

l7_1 = conlayer_right(3,1,12,24)
l7_2 = conlayer_left(3,1,12,12)
l7_3 = conlayer_left(3,1,12,6)

l8_1 = conlayer_right(3,1,6,12)
l8_2 = conlayer_left(3,1,6,6)
l8_3 = conlayer_left(3,1,6,3)

l9_1 = conlayer_right(3,1,3,6)
l9_2 = conlayer_left(3,1,3,3)
l9_3 = conlayer_left(3,1,3,3)

l10_final = conlayer_left(3,1,3,1)

# ---- make graph ----
X1 = tf.placeholder(shape=[None,1024,1,1],dtype=tf.float32)
X2 = tf.placeholder(shape=[None,512,1,1],dtype=tf.float32)

layer11_1 = l1_1.feedforward(X1)
layer11_2 = l1_2.feedforward(layer11_1)
layer11_3 = l1_3.feedforward(layer11_2)

layer21_Input = tf.nn.max_pool(layer11_3,ksize=[1,2,1,1],strides=[1,2,1,1],padding='VALID')
layer21_1 = l2_1.feedforward(layer21_Input)
layer21_2 = l2_2.feedforward(layer21_1)
layer21_3 = l2_3.feedforward(layer21_2)

layer31_Input = tf.nn.max_pool(layer21_3,ksize=[1,2,1,1],strides=[1,2,1,1],padding='VALID')
layer31_1 = l3_1.feedforward(layer31_Input)
layer31_2 = l3_2.feedforward(layer31_1)
layer31_3 = l3_3.feedforward(layer31_2)

layer41_Input = tf.nn.max_pool(layer31_3,ksize=[1,2,1,1],strides=[1,2,1,1],padding='VALID')
layer41_1 = l4_1.feedforward(layer41_Input)
layer41_2 = l4_2.feedforward(layer41_1)
layer41_3 = l4_3.feedforward(layer41_2)

layer51_Input = tf.nn.max_pool(layer41_3,ksize=[1,2,1,1],strides=[1,2,1,1],padding='VALID')
layer51_1 = l5_1.feedforward(layer51_Input)
layer51_2 = l5_2.feedforward(layer51_1)
layer51_3 = l5_3.feedforward(layer51_2)

layer61_Input = tf.concat([layer51_3,layer51_Input],axis=3)
layer61_1 = l6_1.feedforward(layer61_Input)
layer61_2 = l6_2.feedforward(layer61_1)
layer61_3 = l6_3.feedforward(layer61_2)

layer71_Input = tf.concat([layer61_3,layer41_Input],axis=3)
layer71_1 = l7_1.feedforward(layer71_Input)
layer71_2 = l7_2.feedforward(layer71_1)
layer71_3 = l7_3.feedforward(layer71_2)

layer81_Input = tf.concat([layer71_3,layer31_Input],axis=3)
layer81_1 = l8_1.feedforward(layer81_Input)
layer81_2 = l8_2.feedforward(layer81_1)
layer81_3 = l8_3.feedforward(layer81_2)

layer91_Input = tf.concat([layer81_3,layer21_Input],axis=3)
layer91_1 = l9_1.feedforward(layer91_Input)
layer91_2 = l9_2.feedforward(layer91_1)
layer91_3 = l9_3.feedforward(layer91_2)

layer101 = l10_final.feedforward(layer91_3)

#------------- view 2----------------
layer12_1 = l1_1.feedforward(X2)
layer12_2 = l1_2.feedforward(layer12_1)
layer12_3 = l1_3.feedforward(layer12_2)

layer22_Input = tf.nn.max_pool(layer12_3,ksize=[1,2,1,1],strides=[1,2,1,1],padding='VALID')
layer22_1 = l2_1.feedforward(layer22_Input)
layer22_2 = l2_2.feedforward(layer22_1)
layer22_3 = l2_3.feedforward(layer22_2)

layer32_Input = tf.nn.max_pool(layer22_3,ksize=[1,2,1,1],strides=[1,2,1,1],padding='VALID')
layer32_1 = l3_1.feedforward(layer32_Input)
layer32_2 = l3_2.feedforward(layer32_1)
layer32_3 = l3_3.feedforward(layer32_2)

layer42_Input = tf.nn.max_pool(layer32_3,ksize=[1,2,1,1],strides=[1,2,1,1],padding='VALID')
layer42_1 = l4_1.feedforward(layer42_Input)
layer42_2 = l4_2.feedforward(layer42_1)
layer42_3 = l4_3.feedforward(layer42_2)

layer52_Input = tf.nn.max_pool(layer42_3,ksize=[1,2,1,1],strides=[1,2,1,1],padding='VALID')
layer52_1 = l5_1.feedforward(layer52_Input)
layer52_2 = l5_2.feedforward(layer52_1)
layer52_3 = l5_3.feedforward(layer52_2)

layer62_Input = tf.concat([layer52_3,layer52_Input],axis=3)
layer62_1 = l6_1.feedforward(layer62_Input)
layer62_2 = l6_2.feedforward(layer62_1)
layer62_3 = l6_3.feedforward(layer62_2)

layer72_Input = tf.concat([layer62_3,layer42_Input],axis=3)
layer72_1 = l7_1.feedforward(layer72_Input)
layer72_2 = l7_2.feedforward(layer72_1)
layer72_3 = l7_3.feedforward(layer72_2)

layer82_Input = tf.concat([layer72_3,layer32_Input],axis=3)
layer82_1 = l8_1.feedforward(layer82_Input)
layer82_2 = l8_2.feedforward(layer82_1)
layer82_3 = l8_3.feedforward(layer82_2)

layer92_Input = tf.concat([layer82_3,layer22_Input],axis=3)
layer92_1 = l9_1.feedforward(layer92_Input)
layer92_2 = l9_2.feedforward(layer92_1)
layer92_3 = l9_3.feedforward(layer92_2)

layer102 = l10_final.feedforward(layer92_3)

cost = ccaSvd(layer101,layer102,0,0)
auto_train = tf.train.AdamOptimizer(learning_rate=init_lr).minimize(cost)

# --- start session ---
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for iter in range(num_epoch):
    index = np.arange(train_images.shape[0])
    np.random.shuffle(index)
    trX1 = train_X1[index]
    trX2 = train_X2[index]
    # train
    for current_batch_index in range(0,len(train_images),batch_size):
        current_batch_X1 = trX1[current_batch_index:current_batch_index+batch_size,:,:,:]
        current_batch_X2 = trX2[current_batch_index:current_batch_index+batch_size,:,:,:]
        sess_results = sess.run([cost,auto_train],feed_dict={X1:current_batch,X2:current_batch_X2})
        print(" Cost:  %.30f" % sess_results[0], ' Iter: ', iter, '   ', end='\r' )
    print('\n-----------------------')

for data_index in range(0,len(train_images),batch_size):
    current_batch = train_images[data_index:data_index+batch_size,:,:,:]
    current_label = train_labels[data_index:data_index+batch_size,:,:,:]
    sess_results = sess.run(layer6,feed_dict={x:current_batch})

    plt.figure()
    plt.imshow(np.squeeze(current_batch[0,:,:,:]),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"a_Original Image")
    plt.savefig('./gif/'+str(data_index)+"a_Original_Image.png")

    plt.figure()
    plt.imshow(np.squeeze(current_label[0,:,:,:]),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"b_Original Mask")
    plt.savefig('./gif/'+str(data_index)+"b_Original_Mask.png")

    plt.figure()
    plt.imshow(np.squeeze(sess_results[0,:,:,:]),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"c_Generated Mask")
    plt.savefig('./gif/'+str(data_index)+"c_Generated_Mask.png")

    plt.figure()
    plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(current_label[0,:,:,:])),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"d_Original Image Overlay")
    plt.savefig('./gif/'+str(data_index)+"d_Original_Image_Overlay.png")

    plt.figure()
    plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(sess_results[0,:,:,:])),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"e_Generated Image Overlay")
    plt.savefig('./gif/'+str(data_index)+"e_Generated_Image_Overlay.png")

    plt.close('all')


for data_index in range(0,len(test_images),batch_size):
    current_batch = test_images[data_index:data_index+batch_size,:,:,:]
    current_label = test_labels[data_index:data_index+batch_size,:,:,:]
    sess_results = sess.run(layer6,feed_dict={x:current_batch})

    plt.figure()
    plt.imshow(np.squeeze(current_batch[0,:,:,:]),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"a_Original Image")
    plt.savefig('./gif/test_'+str(data_index)+"a_Original_Image.png")

    plt.figure()
    plt.imshow(np.squeeze(current_label[0,:,:,:]),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"b_Original Mask")
    plt.savefig('./gif/test_'+str(data_index)+"b_Original_Mask.png")

    plt.figure()
    plt.imshow(np.squeeze(sess_results[0,:,:,:]),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"c_Generated Mask")
    plt.savefig('./gif/test_'+str(data_index)+"c_Generated_Mask.png")

    plt.figure()
    plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(current_label[0,:,:,:])),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"d_Original Image Overlay")
    plt.savefig('./gif/test_'+str(data_index)+"d_Original_Image_Overlay.png")

    plt.figure()
    plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(sess_results[0,:,:,:])),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"e_Generated Image Overlay")
    plt.savefig('./gif/test_'+str(data_index)+"e_Generated_Image_Overlay.png")

    plt.close('all')


# -- end code --

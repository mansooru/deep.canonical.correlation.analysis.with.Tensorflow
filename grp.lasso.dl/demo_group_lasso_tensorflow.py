# -*- coding: utf-8 -*-

"""
This script shows how to use group LASSO regularization for neural networks
in the TensorFlow library. This type of regularization removes entire neurons
during training, by pushing rows of the weight matrices to be zero simultaneously.

The use of group LASSO inside neural networks is described here:

    Scardapane, S., Comminiello, D., Hussain, A. and Uncini, A., 2017.
    Group sparse regularization for deep neural networks. Neurocomputing, 241, pp.81-89.

A preprint version is available on arXiv: https://arxiv.org/abs/1607.00485.

The original code for the paper was written for Lasagne and Theano and is available here:
    https://bitbucket.org/ispamm/group-lasso-deep-networks

Note that you can combine group LASSO and L1/L2 regularization for better effects.
The most important part of the code are lines 29-37. The function 'group_regularization'
creates the regularization term (from the list of weight matrices), which can
subsequently be added to any error loss (see line 109).

We use the TensorBoard to plot the loss and the number of neurons removed
during training.
"""

import tensorflow as tf

def l21_norm(W):
    # Computes the L21 norm of a symbolic matrix W
    return tf.reduce_sum(tf.norm(W, axis=1))

def group_regularization(v):
    # Computes a group regularization loss from a list of weight matrices corresponding
    # to the different layers (see line 93 for its use).
    const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1], tf.float32))
    return tf.reduce_sum([tf.multiply(const_coeff(W), l21_norm(W)) for W in v if 'bias' not in W.name])

def main():

    # Reset everything
    tf.reset_default_graph()

    # The directory to save TensorBoard summaries
    from datetime import datetime
    now = datetime.now()
    logdir = "summaries/" + now.strftime("%Y%m%d-%H%M%S") + "/"

    # We use a simple regression dataset taken from scikit-learn
    from sklearn import datasets
    data = datasets.load_boston()

    # Preprocess the inputs to be in [-1,1] and split the data in train/test sets
    from sklearn import preprocessing, model_selection
    X = preprocessing.MinMaxScaler(feature_range=(-1,+1)).fit_transform(data['data'])
    y = preprocessing.MinMaxScaler().fit_transform(data['target'].reshape(-1, 1))
    X_trn, X_tst, y_trn, y_tst = model_selection.train_test_split(X, y, test_size=0.25)

    # Placeholders for input and output
    x = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name='input')
    d = tf.placeholder(tf.float32, shape=[None, 1], name='targets')

    # Helper function to generate a layer
    def create_layer(in_var, in_size, out_size):

        # Parameters for input-hidden layer
        W = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[out_size]), name='bias')

        # Output of the hidden layer
        return tf.nn.relu(tf.matmul(in_var, W) + b)

    # We define a simple network with two hidden layers
    with tf.name_scope('hidden_1'):
        h1 = create_layer(x, X.shape[1], 20)
    with tf.name_scope('hidden_2'):
        h2 = create_layer(h1, 20, 15)
    with tf.name_scope('output'):
        y = create_layer(h2, 15, 1)

    # Helper function to check how many neurons are left in a layer
    count_neurons = lambda W: tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(tf.abs(W), reduction_indices=[1]), 10**-3),tf.float32))

    # Get all trainable variables except biases
    v = tf.trainable_variables()
    neurons_summary = tf.summary.scalar('neurons', tf.reduce_sum([count_neurons(W) for W in v if 'bias' not in W.name]))

    # Define the error function
    with tf.name_scope('squared_loss'):
        loss = tf.reduce_mean(tf.squared_difference(d, y))

    # Compute the regularization term
    with tf.name_scope('group_regularization'):
        reg_loss = 0.001*group_regularization(v)

    # We attach a logger to the error loss and the regularization part
    loss_summary = tf.summary.scalar('loss', loss)
    reg_loss_summary = tf.summary.scalar('reg_loss', reg_loss)

    # Merge summaries and write them in output
    merged = tf.summary.merge([loss_summary, reg_loss_summary, neurons_summary])

    with tf.Session() as sess:

        # Initialize the summary writer
        train_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

        with tf.name_scope('train'):
            # Training function
            train_step = tf.train.AdamOptimizer().minimize(tf.add(loss, reg_loss))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        for i in range(1500):

            # Take one training step
            summary, _ = sess.run([merged, train_step], feed_dict={x: X_trn, d: y_trn})
            train_writer.add_summary(summary, i)

        print('Final loss on test set: ', sess.run([loss], feed_dict={x: X_tst, d: y_tst}))

    train_writer.flush()
    train_writer.close()

if __name__ == '__main__':
    main()

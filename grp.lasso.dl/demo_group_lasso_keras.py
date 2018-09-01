# -*- coding: utf-8 -*-

"""
Same setup as 'demo_group_lasso_tensorflow', but we implement training and regularization
inside the Keras library. See the other demo for details. The most important part
of the code is the L21 class (see below), which can be added to any Keras Dense layer.

Note that we do not use TensorBoard here, but we simply plot the loss and number of neurons
obtained from a callback class using Matplotlib.
"""

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import Regularizer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class L21(Regularizer):
    """Regularizer for L21 regularization.
    # Arguments
        C: Float; L21 regularization factor.
    """

    def __init__(self, C=0.):
        self.C = K.cast_to_floatx(C)

    def __call__(self, x):
        const_coeff = np.sqrt(K.int_shape(x)[1])
        return self.C*const_coeff*K.sum(K.sqrt(K.sum(K.square(x), axis=1)))

    def get_config(self):
        return {'C': float(self.l1)}

# Utility function to count active neurons in a Keras model with Dense layers
def count_neurons(model):
    return np.sum([np.sum(np.sum(np.abs(l.get_weights()[0]), axis=1) > 10**-3) \
                          for l in model.layers])

# Callback class to save training loss and the number of neurons
class TrainHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.neurons = [count_neurons(self.model)]

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.neurons.append(count_neurons(self.model))

def main():

    # Reset session
    tf.reset_default_graph()
    K.set_session(tf.Session())

    # We use a simple regression dataset taken from scikit-learn
    from sklearn import datasets
    data = datasets.load_boston()

    # Preprocess the inputs to be in [-1,1] and split the data in train/test sets
    from sklearn import preprocessing, model_selection
    X = preprocessing.MinMaxScaler(feature_range=(-1,+1)).fit_transform(data['data'])
    y = preprocessing.MinMaxScaler().fit_transform(data['target'].reshape(-1, 1))
    X_trn, X_tst, y_trn, y_tst = model_selection.train_test_split(X, y, test_size=0.25)

    # Define the model in Keras
    model = Sequential()
    model.add(Dense(20, input_dim=X.shape[1],  activation='relu', kernel_regularizer=L21(0.001)))
    model.add(Dense(15, activation='relu', kernel_regularizer=L21(0.001)))
    model.add(Dense(1, activation='relu', kernel_regularizer=L21(0.001)))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Get callbacks
    history = TrainHistory()

    # Train using batch gradient descent
    model.fit(X_trn, y_trn, epochs=2500, shuffle=False, batch_size=X_trn.shape[0], verbose=0, callbacks=[history])

    # Evaluate on test data
    y_tst_hat = model.predict(X_tst, batch_size=X_tst.shape[0])
    print('Final loss on test set: ', mean_squared_error(y_tst, y_tst_hat))

    # Plot the training loss during training
    plt.figure()
    plt.plot(history.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.show()

    # Plot the active neurons during training
    plt.figure()
    plt.plot(history.neurons)
    plt.xlabel('Epoch')
    plt.ylabel('Active neurons')
    plt.show()


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-

"""
Simple unit tests to validate the group LASSO penalty for both demos.
"""

import tensorflow as tf
import keras.backend as K
import numpy as np
import unittest
from demo_group_lasso_tensorflow import l21_norm, group_regularization
from demo_group_lasso_keras import L21

class GroupLassoTensorFlowTest(tf.test.TestCase):

    def testL21norm(self):
        # Test the l21_norm function inside the TensorFlow demo.
        with self.test_session():
            x = tf.constant([[0, 0.1], [-0.3, 0.6], [0, 0]])
            l21norm = l21_norm(x)
            real_l21norm = np.sqrt(0.1*0.1) + np.sqrt(0.3*0.3 + 0.6*0.6)
            self.assertAlmostEqual(l21norm.eval(), real_l21norm, places=5)

    def testGroupLassoRegularization(self):
        # Test the group_regularization function inside the TensorFlow demo.
        with self.test_session():
            W1 = tf.constant([[0, 0.1], [-0.3, 0.6], [0, 0]])
            W2 = tf.constant([[0.3, -0.2, 0], [0.1, -0.1, 0.2]])
            l21_norm_W1 = l21_norm(W1).eval()
            l21_norm_W2 = l21_norm(W2).eval()
            self.assertAlmostEqual(group_regularization((W1, W2)).eval(), \
                                   np.sqrt(2)*l21_norm_W1 + np.sqrt(3)*l21_norm_W2,
                                          places=5)

class GroupLassoKerasTest(unittest.TestCase):

    def testL21norm(self):
        # Test the L21 class inside the Keras demo.
        tf.reset_default_graph()
        K.set_session(tf.Session())
        x = np.asarray([[0, 0.1], [-0.3, 0.6], [0, 0]])
        x_keras = K.variable(x)
        L21_reg = L21(1.0)
        l21_norm_x = L21_reg(x_keras)
        self.assertAlmostEqual(l21_norm_x.eval(session=K.get_session()), \
                               np.sqrt(2)*(np.sqrt(0.1*0.1) + np.sqrt(0.3*0.3 + 0.6*0.6)),\
                                    places=5)

if __name__ == '__main__':
    tf.test.main()

import numpy
import tensorflow as tf

def neg_correlation(output1, output2,outdim_size):
    r1 = 1e-4
    r2 = 1e-4
    eps = 1e-12

    # unpack (separate) the output of networks for view 1 and view 2
    H1 = tf.transpose(output1)
    H2 = tf.transpose(output2)

    m = tf.shape(H1)[1]

    H1bar = H1 - (1.0 / tf.cast(m, tf.float32)) * tf.matmul(H1, tf.ones([m, m]))
    H2bar = H2 - (1.0 / tf.cast(m, tf.float32)) * tf.matmul(H2, tf.ones([m, m]))

    SigmaHat12 = (1.0 / (tf.cast(m, tf.float32) - 1)) * tf.matmul(H1bar, tf.transpose(H2bar))
    SigmaHat11 = (1.0 / (tf.cast(m, tf.float32) - 1)) * tf.matmul(H1bar, tf.transpose(H1bar)) + r1 * tf.eye(outdim_size)
    SigmaHat22 = (1.0 / (tf.cast(m, tf.float32) - 1)) * tf.matmul(H2bar, tf.transpose(H2bar)) + r2 * tf.eye(outdim_size)

    # Calculating the root inverse of covariance matrices by using eigen decomposition
    [D1, V1] = tf.linalg.eigh(SigmaHat11)
    [D2, V2] = tf.linalg.eigh(SigmaHat22)

    # Added to increase stability
    posInd1 = tf.where(tf.greater(D1, eps))
    posInd1 = tf.reshape(posInd1, [-1, tf.shape(posInd1)[0]])[0]
    D1 = tf.gather(D1, posInd1)
    V1 = tf.gather(V1, posInd1)

    posInd2 = tf.where(tf.greater(D2, eps))
    posInd2 = tf.reshape(posInd2, [-1, tf.shape(posInd2)[0]])[0]
    D2 = tf.gather(D2, posInd2)
    V2 = tf.gather(V2, posInd2)

    SigmaHat11RootInv = tf.matmul(tf.matmul(V1, tf.linalg.diag(D1 ** -0.5)), tf.transpose(V1))
    SigmaHat22RootInv = tf.matmul(tf.matmul(V2, tf.linalg.diag(D2 ** -0.5)), tf.transpose(V2))

    Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    # all singular values are used to calculate the correlation
    # corr = tf.sqrt(tf.linalg.trace(tf.matmul(tf.transpose(Tval), Tval)))  ### The usage of "sqrt" here is wrong!!!
    Tval.set_shape([outdim_size, outdim_size])
    s = tf.svd(Tval, compute_uv=False)
    corr = tf.reduce_sum(s)

    return -corr

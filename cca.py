import tensorflow as tf

def ccaSvd(XArr,YArr,lambda1,lambda2):
     dataMatX = XArr
     dataMatY = YArr
     n = tf.shape(dataMatX)[0]
     p = tf.shape(dataMatX)[1]
     q = tf.shape(dataMatY)[1]
     u0 = tf.random_normal([p,1],stddev=0.01)
     v0 = tf.random_normal([q,1],stddev=0.01)
     k=1
     dataMatXMean = dataMatX - tf.ones([n, 1]) * tf.reduce_mean(dataMatX, 0)
     dataMatYMean = dataMatY - tf.ones([n, 1]) * tf.reduce_mean(dataMatY, 0)
     K = tf.matmul(tf.transpose(dataMatXMean),dataMatYMean)
     for i in range(1,100):
         u = tf.matmul(K,v0);
         if (tf.norm(u,ord=2) != 0):
             u = tf.div(u, tf.norm(u,ord=2))
         tmp_u ,_ = tf.nn.top_k(tf.transpose(tf.abs(u)),p)
         th_u = tmp_u[0,lambda1-1]
         u = tf.sign(u) * tf.maximum(tf.abs(u)-th_u*tf.ones([p,1]),0.0)
         if (tf.norm(u,ord=2) != 0):
             u = tf.div(u,tf.norm(u,ord=2))

         v = tf.matmul(tf.transpose(K),u)
         if (tf.norm(v,ord=2) != 0):
             v = tf.div(v,tf.norm(v,ord=2))
         tmp_v ,_ = tf.nn.top_k(tf.transpose(tf.abs(v)),q-1)
         th_v = tmp_v[0,lambda2-1]
         v = tf.sign(v) * tf.maximum(tf.abs(v) - th_v*tf.ones([q,1]),0.0)
         if (tf.norm(v,ord=2) != 0):
             v = tf.div(v,tf.norm(v,ord=2))
#         if tf.less(tf.reduce_mean(tf.square(v-v0)), 0.001):
#              corr = tf.matmul(tf.transpose(tf.matmul(dataMatXMean,u)),tf.matmul(dataMatYMean,v))
#              return -corr
#         v0=v;
         k=k+1
     corr = tf.matmul(tf.transpose(tf.matmul(dataMatXMean,u)),tf.matmul(dataMatYMean,v))
     return -corr

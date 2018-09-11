import tensorflow as tf

def l21_norm(W):
    # Computes the L21 norm of a symbolic matrix W
    return tf.reduce_sum(tf.norm(W, axis=1))

def group_regularization(v):
    # Computes a group regularization loss from a list of weight matrices corresponding
    # to the different layers (see line 93 for its use).
    const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1], tf.float32))
    return tf.reduce_sum([tf.multiply(const_coeff(W), l21_norm(W)) for W in v if 'bias' not in W.name])


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

# We define a simple network with two hidden layers
W1 = tf.Variable(tf.truncated_normal([X.shape[1], 20], stddev=0.1),name='W')
b1 = tf.Variable(tf.constant(0.1, shape=[20]), name='bias')
L1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([20, 13], stddev=0.1),name='W')
b2 = tf.Variable(tf.constant(0.1, shape=[13]), name='bias')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([13, 1], stddev=0.1),name='W')
b3 = tf.Variable(tf.constant(0.1, shape=[1]), name='bias')
y = tf.nn.relu(tf.matmul(L2, W3) + b3)

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

# Initialize the summary writer
train_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
cost = tf.add(loss, reg_loss)
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Initialize all variables
batch_size = 50
n_epochs = 1500
for epoch in range(n_epochs):
    index = np.arange(X_trn.shape[0])
    np.random.shuffle(index)
    trX1 = X_trn[index]
    trX2= y_trn[index]

    train_cost = 0
    for current_batch_index in range(0,len(X_trn),batch_size):
        current_batch_X1 = trX1[current_batch_index:current_batch_index+batch_size,:]
        current_batch_X2 = trX2[current_batch_index:current_batch_index+batch_size,:]

        summary, _  = sess.run([merged, optimizer], feed_dict={x:current_batch_X1, d:current_batch_X2})
        train_writer.add_summary(summary, epoch)
        training_loss = sess.run(cost, feed_dict={x:current_batch_X1, d:current_batch_X2})
        train_cost = training_loss + train_cost

    train_cost = train_cost / (len(X_trn)/batch_size)
    test_cost = sess.run(cost,feed_dict={x:X_tst,d:y_tst})
    print("iteration:", epoch, "loss_for_train:", train_cost,"loss_for_tune:", test_cost, end='\r')



train_writer.flush()
train_writer.close()

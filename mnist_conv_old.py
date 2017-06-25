#!/usr/bin/env python

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from EUNN_semiunitary import *

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def num_repeats(n, m):
    return int((m*(n-m) + m*(m-1)/2) / ((n/2) * np.log2(n)))

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1,28,28,1])

# with tf.variable_scope("test"):
#     test = EUNN_rect(ops.convert_to_tensor(np.eye(5 * 5 * 1), dtype=tf.float32), [5 * 5 * 1, 32])
#     print(test.shape)

# Layer 1 Conv
with tf.variable_scope("conv1"):
    # W_conv1 = tf.reshape(tf.matrix_transpose(EUNN(ops.convert_to_tensor(np.eye(32, 6 * 6 * 1), dtype=tf.float32)
    #                                             )), [6, 6, 1, 32])
    W_conv1 = tf.reshape(EUNN_rect(ops.convert_to_tensor(np.eye(6 * 6 * 1), dtype=tf.float32)
                                    , [6 * 6 * 1, 32]), [6, 6, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

# Layer 2 Conv
with tf.variable_scope("conv2"):
    # W_conv2 = tf.reshape(tf.matrix_transpose(EUNN(ops.convert_to_tensor(np.eye(64, 5 * 5 * 32), dtype=tf.float32)
    #                                             )), [5, 5, 32, 64])
    W_conv2 = tf.reshape(EUNN_rect(ops.convert_to_tensor(np.eye(5 * 5 * 32), dtype=tf.float32)
                                    , [5 * 5 * 32, 64]), [5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# Layer 3 Fully Connected
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Layer 4 Readout
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# W_conv1_reporter = tf.reshape(W_conv1, [6 * 6 * 1, 32])

sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%50 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        # print(W_conv1_reporter.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0}))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))













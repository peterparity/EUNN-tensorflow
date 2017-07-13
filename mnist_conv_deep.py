#!/usr/bin/env python

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from EUNN_semiunitary import *

# sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d_full(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv2d(x, ksize, num_kernels):
    # input_shape = x.get_shape().as_list()
    # depth = input_shape[-1]
    depth = int(x.get_shape()[-1])

    image_patches = tf.extract_image_patches(x, ksizes=[1, ksize[0], ksize[1], 1], strides=[1, 1, 1, 1],
                                                rates=[1, 1, 1, 1], padding='SAME')
    # image_patches = tf.nn.conv2d(x, tf.reshape(tf.eye(ksize[0] * ksize[1] * depth),
    #                                             [ksize[0], ksize[1], depth, ksize[0] * ksize[1] * depth]),
    #                                             strides=[1, 1, 1, 1], padding='SAME')
    output = EUNN_rect(tf.reshape(image_patches, [-1, ksize[0] * ksize[1] * depth]),
                                    [ksize[0] * ksize[1] * depth, num_kernels], use_hybrid_method=True)

    # return tf.reshape(output, [-1] + input_shape[1:3] + [num_kernels])
    return output

def feed_conv2d(x, ksize, num_kernels, batch_size):
    input_shape = x.get_shape().as_list()
    x_length = tf.shape(x)[0]

    with tf.control_dependencies([tf.assert_equal(tf.floormod(x_length, batch_size), 0)]):
        x = toTensorArray(x)

        output = tf.TensorArray(dtype=x.dtype, 
                                size=x_length/batch_size,
                                dynamic_size=False,
                                infer_shape=True)

        i = 0

        def feed(output, i):
            current_batch = tf.stack([x.read(i * batch_size + j) for j in range(batch_size)])
            output = output.write(i, conv2d(current_batch, ksize, num_kernels))
            i += 1

            return output, i

        def cond(output, i):
            return tf.less(i, x_length/batch_size)

        loop_vars = [output, i]
        output, _ = tf.while_loop(
            cond, 
            feed, 
            loop_vars
        )

    return tf.reshape(output.stack(), [-1] + input_shape[1:3] + [num_kernels])

def conv_layer(input_images, ksize, num_kernels, name, batch_size=50, maxpool=True, full=False):
    with tf.variable_scope(name):
        b_conv = bias_variable([num_kernels])

        if full:
            depth = int(input_images.get_shape()[-1])
            W_conv = weight_variable([ksize[0], ksize[1], depth, num_kernels])
            h_conv = tf.nn.relu(conv2d_full(input_images, W_conv) + b_conv)
        else:
            h_conv = tf.nn.relu(feed_conv2d(input_images, ksize, num_kernels, batch_size) + b_conv)
        
        if maxpool:
            output = max_pool_2x2(h_conv)
        else:
            output = h_conv

    return output


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

current_images = x_image
for i in range(5):
    maxpool = False
    if i >= 3: maxpool = True
    current_images = conv_layer(current_images, [5, 5], 2**(i+1), "conv_" + str(i), 
                                    batch_size=50, maxpool=maxpool, full=False)

# Layer 3 Fully Connected
with tf.name_scope("full1"):
    W_fc1 = weight_variable([7 * 7 * 32, 1024])
    b_fc1 = bias_variable([1024])

    current_images_flat = tf.reshape(current_images, [-1, 7 * 7 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(current_images_flat, W_fc1) + b_fc1)

# Dropout
with tf.name_scope("dropout"):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Layer 4 Readout
with tf.name_scope("output"):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# writer = tf.summary.FileWriter("/Users/peter/Dropbox (MIT)/Soljacic/semiunitaryConvNet/tmp/1")
# writer.add_graph(sess.graph)

# Run Training
for i in range(12000):
    batch = mnist.train.next_batch(50)
    if i%50 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    # print(sess.run(tf.shape(x_image), feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))













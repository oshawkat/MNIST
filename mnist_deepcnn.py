# Build a deep convolutional neural network for identifying written digits [0, 9]
# Follows along withthe "Deep MNIST for Experts" tutorial on the TensorFlow website
# https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf

# Functions for prettier weight initialization
def weight_variable(shape):
	# initialize with a small amount of noise for symmetry breaking and 
	# preventing 0 gradient
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	# Slight positive bias to avoid dead neurons
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

# Functions for prettier convolution and pooling
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


# Downloads MNIST data
from tensorflow.examples.tutorials.mnist import input_data	
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Input data
x = tf.placeholder(tf.float32, [None, 784])	# Images come flattended from 28x28px
x_image = tf.reshape(x, [-1, 28, 28, 1])	# reshape into 28x28px images, single color chan
y_truth = tf.placeholder(tf.float32, [None, 10])	# ground truth

# First Convolutional layer.  Will compute 32 features for each 5x5 patch
W_conv1 = weight_variable([5, 5, 1, 32])	# patch size (row, col), input chanels, output chanels
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)	# convolve, add bias, and apply ReLU
h_pool1 = max_pool_2x2(h_conv1)	# Reduces image size to 14x14

# Second concolutional layer - Compute 64 features for each 5x5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)	# Reduces image to 7x7

# Fully Connected layer with 1024 neurons for processing entire image
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# reshape pooling layer outputinto a batch of vectors
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# Multiply by pooling by weight matrix, add bias, and apply ReLU
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Apply a dropout before the readout layer to avoid overfitting
# Use a placeholder for prob that a neuron is kept to turn 
# dropout on during training and off for testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer 
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and Evaluate Efficacy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_truth, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)	# Use Adam (vs max grad decent)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_truth, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run the actual session.  Using 'with' ensures that it is auto destroyed
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict = {
				x: batch[0], y_truth: batch[1], keep_prob: 1.0})
			print("Step %d,\t training accuracy %g" % (i, train_accuracy))
		train_step.run(feed_dict = {x: batch[0], y_truth: batch[1], keep_prob: 0.5})

	print("Test accuracy %g" % (accuracy.eval(feed_dict = {
		x: mnist.test.images, y_truth: mnist.test.labels, keep_prob: 1.0})))


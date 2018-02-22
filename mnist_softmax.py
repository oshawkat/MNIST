# Follows along withthe "MNIST for ML Beginners" tutorial on the TensorFlow website
# https://www.tensorflow.org/get_started/mnist/beginners

import tensorflow as tf


# Downloads MNIST data
from tensorflow.examples.tutorials.mnist import input_data	
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Will be modeling using softmax: y = softmax(Wx + b)
x = tf.placeholder(tf.float32, [None, 784])	# Images are flattended from 28x28px
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Defining Loss
y_truth = tf.placeholder(tf.float32, [None, 10])	# ground truth
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_truth, logits=y))

# Training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Run session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
	batch_x, batch_y = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict = {x: batch_x, y_truth: batch_y})

# Evaluate Model efficacy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_truth, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print (sess.run(accuracy, feed_dict = {x: mnist.test.images, y_truth: mnist.test.labels}))


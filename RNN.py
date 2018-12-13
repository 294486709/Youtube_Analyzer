import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from random import shuffle

# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#
# '''
# To classify images using a recurrent neural network, we consider every image
# row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
# handle 28 sequences of 28 steps for every sample.
# '''
#
# # Training Parameters
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.80
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
learning_rate = 0.001
num_epoch = 8
batch_size = 128
display_step = 10
#
# # Network Parameters
num_input = 300 # MNIST data input (img shape: 28*28)
timesteps = 30 # timesteps
num_hidden = 1024 # hidden layer num of features
num_classes = 6 # MNIST total classes (0-9 digits)
#
# # tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input],name='X')
Y = tf.placeholder("float", [None, num_classes],name='Y')
#
# # Define weights
weights = {
	'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
	'out': tf.Variable(tf.random_normal([num_classes]))
}
#
#
def RNN(x, weights, biases):

	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, timesteps, n_input)
	# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

	# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
	x = tf.unstack(x, timesteps, 1)

	# Define a lstm cell with tensorflow
	lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0)

	# Get lstm cell output
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

	# Linear activation, using rnn inner loop last output
	ooo = tf.add(tf.matmul(outputs[-1], weights['out']) , biases['out'], name='outt')
	YYY = tf.nn.softmax(ooo, name='Y1')
	return ooo

def next_batch(idx, xtrain,ytrian):
	current = idx*batch_size
	idx = current + batch_size
	resx = xtrain[current:idx]
	resy = ytrian[current:idx]
	return resx, resy


def get_eval(n,xevl,yevl):
	com = list(zip(xevl, yevl))
	shuffle(com)
	xevl[:], yevl[:] = zip(*com)
	resx = xevl[:n]
	resy = yevl[:n]
	return  resx,resy

def train(X, xtrain, ytrain,xval,yval, xtest, ytest,Y):

	logits = RNN(X, weights, biases)
	prediction = tf.nn.softmax(logits)

	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
		logits=logits, labels=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# Evaluate model (with test logits, for dropout to be disabled)
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	saver = tf.train.Saver()
	# Start training
	with tf.Session(config=config) as sess:

		# Run the initializer
		sess.run(init)

		for epoch in range(num_epoch):
			print('epoch',epoch)
			com = list(zip(xtrain, ytrain))
			shuffle(com)
			xtrain[:], ytrain[:] = zip(*com)
			loss = 0
			for idx in range(int(len(xtrain) / batch_size)):
				batch_x, batch_y = next_batch(idx, xtrain, ytrain)
			# batch_x, batch_y = mnist.train.next_batch(batch_size)
			# Reshape data to get 28 seq of 28 elements
				batch_x = np.array(batch_x).reshape((batch_size, timesteps, num_input))
				# Run optimization op (backprop)
				sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
				if idx % display_step == 0 or idx == 1:
					# Calculate batch loss and accuracy
					loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
																		 Y: batch_y})
					print("Step " + str(epoch) + ", Minibatch Loss= " + \
						  "{:.4f}".format(loss) + ", Training Accuracy= " + \
						  "{:.3f}".format(acc))

		print("Optimization Finished!")
		saver.save(sess, "Rnn/result.ckpt")
		total = 0
		for j in range(10):
			com = list(zip(xtest, ytest))
			shuffle(com)
			xtest[:], ytest[:] = zip(*com)
			test_len = 1280
			test_data = np.array(xtest)[:test_len].reshape((-1, timesteps, num_input))
			test_label = ytest[:test_len]
			tempacc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
			total += float(tempacc)
			print("Testing Accuracy:", tempacc)
		print('Final Acc:',total/10)


def main():
	xtrain = list(np.load('dataset/xtrain.npy'))
	# xtrain = reshape(xtrain)
	ytrain = list(np.load('dataset/ytrain.npy'))
	xtest = list(np.load('dataset/xtest.npy'))
	# xtest = reshape(xtest)
	ytest = list(np.load('dataset/ytest.npy'))
	xval = list(np.load('dataset/xval.npy'))
	# xval = reshape(xval)
	yval = list(np.load('dataset/yval.npy'))
	train(X, xtrain, ytrain,xval,yval, xtest, ytest,Y)


if __name__ == '__main__':
	main()

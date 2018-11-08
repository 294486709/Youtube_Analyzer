import tensorflow as tf
import numpy as np
from random import shuffle
from tqdm import tqdm

NUM_Class = 6
BATCH_SIZE = 200
DATA_SIZE = 30
fc_size = 32*15*15
NUM_EPOCH = 10

x = tf.placeholder('float', [None, DATA_SIZE,DATA_SIZE],name='X')
y = tf.placeholder('float', [None, NUM_Class],'Y')


def CNN(x):
	weights = {'w_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
			   # 'w_conv2': tf.Variable(tf.random_normal([3,3,30,60])),
			   # 'w_conv3':tf.Variable(tf.random_normal([5,5,64,128])),
			   # 'w_conv4':tf.Variable(tf.random_normal([3,3,128,256])),
			   # 'w_conv5':tf.Variable(tf.random_normal([3,3,256,512])),
			   'w_fc1': tf.Variable(tf.random_normal([fc_size,1024])),
			   # 'w_fc2': tf.Variable(tf.random_normal([1024, 1024])),
			   # 'w_fc3': tf.Variable(tf.random_normal([1024, 1024])),
			   'out': tf.Variable(tf.random_normal([1024,NUM_Class]))}
	biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
			   # 'b_conv2': tf.Variable(tf.random_normal([60])),
			  # 'b_conv3': tf.Variable(tf.random_normal([128])),
			  # 'b_conv4': tf.Variable(tf.random_normal([256])),
			  # 'b_conv5': tf.Variable(tf.random_normal([512])),
			   'b_fc1': tf.Variable(tf.random_normal([1024])),
			  # 'b_fc2':tf.Variable(tf.random_normal([1024])),
			  # 'b_fc3':tf.Variable(tf.random_normal([1024])),
			   'out': tf.Variable(tf.random_normal([NUM_Class]))}
	x = tf.reshape(x, shape=[-1,DATA_SIZE,DATA_SIZE,1])
	conv1 = tf.nn.relu(tf.nn.conv2d(x,weights['w_conv1'],strides=[1,1,1,1], padding='SAME') + biases['b_conv1'])
	conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding= 'SAME')
	#
	# conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1,1,1,1], padding='VALID') + biases['b_conv2'])
	# conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	# conv3 = tf.nn.relu(tf.nn.conv2d(conv2,weights['w_conv3'],strides=[1,1,1,1], padding='SAME') + biases['b_conv3'])
	# conv3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1],padding= 'SAME')
	#
	# conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weights['w_conv4'], strides=[1,1,1,1], padding='VALID') + biases['b_conv4'])
	# conv4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	#
	# conv5 = tf.nn.relu(tf.nn.conv2d(conv4, weights['w_conv5'], strides=[1,1,1,1], padding='VALID') + biases['b_conv5'])
	# conv5 = tf.nn.max_pool(conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



	fc1 = tf.reshape(conv1, [-1,fc_size])
	fc1 = tf.nn.relu(tf.matmul(fc1, weights['w_fc1']) + biases['b_fc1'])

	output = tf.matmul(fc1, weights['out']) + biases['out']
	YYY = tf.nn.softmax(output, name='Y1')
	return output


def next_batch(idx, xtrain,ytrian):
	current = idx*BATCH_SIZE
	idx = current + BATCH_SIZE
	resx = xtrain[current:idx]
	resy = ytrian[current:idx]
	return resx, resy

def train(x,xtrain,ytrain, xval, yval,xtest,ytest, y):
	prediction = CNN(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y),name='cost')
	optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(NUM_EPOCH):
			com = list(zip(xtrain, ytrain))
			shuffle(com)
			xtrain[:], ytrain[:] = zip(*com)
			epoch_loss = 0
			for idx in tqdm(range(int(len(xtrain)/BATCH_SIZE))):
				epoch_x, epoch_y = next_batch(idx,xtrain,ytrain)
				_, c = sess.run([optimizer,cost], feed_dict={x:epoch_x,y:epoch_y})
				epoch_loss += c
				#print("Epoch:{} completed out of {}, loss{}".format(epoch, NUM_EPOCH, epoch_loss))
				correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
				accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			print('Val Accuracy:{}, loss{}, epoch{}'.format(accuracy.eval({x:xval,y:yval}),epoch_loss,epoch))
		print('Test Accuracy:{}, loss{}, epoch{}'.format(accuracy.eval({x: xtest, y: ytest}), epoch_loss, epoch))
		save_path = saver.save(sess,"Cnn/result.ckpt")
		print('saved!')




def main():
	xtrain = list(np.load('xtrain.npy'))
	for i in range(len(xtrain)):
		xtrain[i] = np.reshape(xtrain[i],[30,30])
	ytrain = list(np.load('ytrain.npy'))
	xtest = list(np.load('xtest.npy'))
	for i in range(len(xtest)):
		xtest[i] = np.reshape(xtest[i],[30,30])
	ytest = list(np.load('ytest.npy'))
	xval = list(np.load('xval.npy'))
	for i in range(len(xval)):
		xval[i] = np.reshape(xval[i],[30,30])
	yval = list(np.load('yval.npy'))
	train(x, xtrain, ytrain, xval, yval, xtest, ytest,y)


if __name__ == '__main__':
    main()

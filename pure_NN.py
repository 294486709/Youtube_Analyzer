import tensorflow as tf
import numpy as np
from tqdm import tqdm
from random import shuffle
input_size = 100*19
# stationary
n_classes = 6
# changeable
nc_layer1 = 1000
nc_layer2 = 1000
nc_layer3 = 1000
nc_layer4 = 1000
nc_layer5 = 1000
nc_layer6 = 1000
nc_layer7 = 1000
nc_layer8 = 1000
nc_layer9 = 1000
nc_layer10 = 1000
batch_size = 100
x = tf.placeholder('float', [None, input_size],name='x')
y = tf.placeholder('float', [None, n_classes], name='y')
num_epochs = 7


def neural_network_model(data):
	layer1 = {'weights':tf.Variable(tf.random_normal([input_size, nc_layer1]),name='l1w'),'biases':tf.Variable(tf.random_normal([nc_layer1]),name='l1b')}
	layer2 = {'weights':tf.Variable(tf.random_normal([nc_layer1,nc_layer2]),name='l2w'), 'biases':tf.Variable(tf.random_normal([nc_layer2]),name='l2b')}
	layer3 = {'weights':tf.Variable(tf.random_normal([nc_layer2,nc_layer3]),name='l3w'), 'biases':tf.Variable(tf.random_normal([nc_layer3]),name='l3b')}
	layer4 = {'weights':tf.Variable(tf.random_normal([nc_layer3,nc_layer4]),name='l4w'), 'biases':tf.Variable(tf.random_normal([nc_layer4]),name='l4b')}
	layer5 = {'weights':tf.Variable(tf.random_normal([nc_layer4,nc_layer5]),name='l5w'), 'biases':tf.Variable(tf.random_normal([nc_layer5]),name='l5b')}
	layer6 = {'weights':tf.Variable(tf.random_normal([nc_layer5,nc_layer6]),name='l6w'), 'biases':tf.Variable(tf.random_normal([nc_layer6]),name='l6b')}
	layer7 = {'weights':tf.Variable(tf.random_normal([nc_layer6,nc_layer7]),name='l7w'), 'biases':tf.Variable(tf.random_normal([nc_layer7]),name='l7b')}
	layer8 = {'weights':tf.Variable(tf.random_normal([nc_layer7,nc_layer8]),name='l8w'), 'biases':tf.Variable(tf.random_normal([nc_layer8]),name='l8b')}
	layer9 = {'weights':tf.Variable(tf.random_normal([nc_layer8,nc_layer9]),name='l9w'), 'biases':tf.Variable(tf.random_normal([nc_layer9]),name='l9b')}
	layer10 = {'weights':tf.Variable(tf.random_normal([nc_layer9,nc_layer10]),name='l10w'), 'biases':tf.Variable(tf.random_normal([nc_layer10]),name='l10b')}
	fc_layer1 = {'weights':tf.Variable(tf.random_normal([nc_layer10,n_classes]),name='fcw'), 'biases':tf.Variable(tf.random_normal([n_classes]),name='fcb')}

	l1 = tf.add(tf.matmul(data, layer1['weights']), layer1['biases'],name='l1')
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'],name='l2')
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, layer3['weights']), layer3['biases'],name='l3')
	l3 = tf.nn.relu(l3)

	l4 = tf.add(tf.matmul(l3, layer4['weights']), layer4['biases'],name='l4')
	l4 = tf.nn.relu(l4)

	l5 = tf.add(tf.matmul(l4, layer5['weights']), layer5['biases'],name='l5')
	l5 = tf.nn.relu(l5)

	l6 = tf.add(tf.matmul(l5, layer6['weights']), layer6['biases'],name='l6')
	l6 = tf.nn.relu(l6)

	l7 = tf.add(tf.matmul(l6, layer7['weights']), layer7['biases'],name='l7')
	l7 = tf.nn.relu(l7)

	l8 = tf.add(tf.matmul(l7, layer8['weights']), layer8['biases'],name='l8')
	l8 = tf.nn.relu(l8)

	l9 = tf.add(tf.matmul(l8, layer9['weights']), layer9['biases'],name='l9')
	l9 = tf.nn.relu(l9)

	l10 = tf.add(tf.matmul(l9, layer10['weights']), layer10['biases'],name='l10')
	l10 = tf.nn.relu(l10)

	fc = tf.add(tf.matmul(l10,  fc_layer1['weights']), fc_layer1['biases'],name='fc')
	YYY = tf.nn.softmax(fc, name='Y1')
	return fc


def next_batch(idx, xtrain,ytrian):
	current = idx*batch_size
	idx = current + batch_size
	resx = xtrain[current:idx]
	resy = ytrian[current:idx]
	return resx, resy



def train_NN(x,xtrain,ytrain,xval, yval,xtest,ytest,y):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y), name='cost')
	# learning rate = 0.0001
	optimizer = tf.train.AdamOptimizer(learning_rate=0.0001,name='optimizer').minimize(cost)
	saver = tf.train.Saver()


	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(num_epochs):
			com = list(zip(xtrain, ytrain))
			shuffle(com)
			xtrain[:], ytrain[:] = zip(*com)
			epoch_loss = 0
			for idx in tqdm(range(int(len(xtrain)/batch_size))):
				#epoch_x, epoch_y = tf.train.batch([xtrain,ytrain],batch_size=batch_size,allow_smaller_final_batch=False)
				epoch_x, epoch_y = next_batch(idx, xtrain,ytrain)
				_, c = sess.run([optimizer,cost], feed_dict={x:epoch_x, y:epoch_y})
				epoch_loss += c

			print("Epoch:{} completed out of {}, loss:{}".format(epoch,num_epochs,epoch_loss))

			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			print('Val Accuracy:{}'.format(accuracy.eval({x:xval,y:yval})))
		print('TEST Accuracy:{}'.format(accuracy.eval({x: xtest, y: ytest})))
		sp = saver.save(sess, "w2vnn/result.ckpt")

def main():
	xtrain = list(np.load('xtrain.npy'))
	ytrain = list(np.load('ytrain.npy'))
	xtest = list(np.load('xtest.npy'))
	ytest = list(np.load('ytest.npy'))
	xval = list(np.load('xval.npy'))
	yval = list(np.load('yval.npy'))
	train_NN(x, xtrain, ytrain, xval, yval, xtest, ytest,y)
if __name__ == '__main__':
	main()








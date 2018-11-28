import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import tensorflow as tf

NUM_FEATRUE = 30

# title is a string
def convert(title):
	title = simple_preprocess(title)
	result = np.zeros([1,NUM_FEATRUE])
	if len(title) > 30:
		title = title[:30]
	model = Word2Vec.load("w2vmodel.w2v")
	for i in title:
		try:
			result = np.vstack([result, model[i]])
		except:
			pass
	result = result[1:]
	if len(result) < 30:
		temp = np.zeros([1,NUM_FEATRUE])
		for i in range(30-len(result)):
			result = np.vstack([result, temp])
	pass
	return  result


def predict(title):
	#title = input("Please input a string:\n")
	#title = "Benches clear twice between Yankees, Red Sox"

		title = convert(title)
		tf.reset_default_graph()
		counter = 0
		with tf.Session() as sess:
			saver = tf.train.import_meta_graph("Cnn/result.ckpt.meta")
			saver.restore(sess, tf.train.latest_checkpoint("Cnn"))
			graph = tf.get_default_graph()
			X = graph.get_tensor_by_name('X:0')
			Y = graph.get_tensor_by_name('Y1:0')

			j = np.reshape(title,[1,30,30])
			result = sess.run(Y, feed_dict={X : j})
			# result = tf.nn.softmax(result)
			# ans = tf.arg_max(result)
			for i in range(len(result[0])):
				if result[0][i] != 0:
					print(i)
					if i == 0: return 'Film&Comedy'
					elif i == 1: return 'Music'
					elif i == 2: return 'Entertainment'
					elif i == 3: return 'Howto&Style'
					elif i == 4: return 'Science&Education'
					else: return 'Sports&Gaming'




#if __name__ == '__main__':
	#predict("President Bill Clinton On Dictators, Democracy, & Why We Need Immigrants More Than Ever")
 #   main()

#convert(title)

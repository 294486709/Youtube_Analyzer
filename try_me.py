import numpy as np
import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NUM_FEATRUE = 300

# title is a string
def convert(title):
    # tokenize and delete excess information
    title = simple_preprocess(title)
    # makeing it 30 hy 30
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
    # padding with 0
    if len(result) < 30:
        temp = np.zeros([1,NUM_FEATRUE])
        for i in range(30-len(result)):
            result = np.vstack([result, temp])
    return  result


def main():
    while(1):
        title = input("Please input a string:")
        # title = "Tensorflow"
        j = convert(title)
        j = np.reshape(j,[1,30,300])
        # testing
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("Rnn/result.ckpt.meta")
            saver.restore(sess, tf.train.latest_checkpoint("Rnn"))
            graph = tf.get_default_graph()
            X = graph.get_tensor_by_name('X:0')
            Y = graph.get_tensor_by_name('outt:0')
            result = sess.run(Y, feed_dict={X : j})
            # result = tf.nn.softmax(result)
            # ans = tf.arg_max(result)
            maxindex  = list(result[0]).index(max(list(result[0])))
            if maxindex == 0:
                print("Movie and Comedy")
            elif maxindex == 1:
                print("Music")
            elif maxindex == 2:
                print("Entertainments")
            elif maxindex == 3:
                print("How to and Style")
            elif maxindex == 4:
                print("Science and Education")
            else:
                print("Sports")






if __name__ == '__main__':
    main()

#convert(title)

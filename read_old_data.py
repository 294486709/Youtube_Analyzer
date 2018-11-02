import nltk
from vocab import Vocab
import os
import codecs
import numpy as np
import random
import pandas as pd
import multiprocessing as mp
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
LENGTH_TRAINING = 80
TESTING_PERSENTATGE = 0.10
num_features = 100
MAX_LENGTH=19

def read_from_file():
	f = pd.read_csv('USvideos_sorted.csv')
	x = list(f[f.columns[2]])
	y = list(f[f.columns[4]])
	return x,y


def main():
	current,label = read_from_file()
	for i in range(len(label)):
		if label[i] in ['0','1','2','3','4','5']:
			f = open(str(label[i]) + '.txt','a')
			f.write(current[i])
			f.write('\n')
			f.close()
	pass



if __name__ == '__main__':
    main()
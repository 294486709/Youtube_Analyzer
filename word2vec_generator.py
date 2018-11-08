# Copyright Yixue Zhang jedzhang@bu.edu
import multiprocessing as mp
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


LENGTH_TRAINING = 80
TESTING_PERSENTATGE = 0.10
num_features = 30


def get_data_file():
	useful_list = [0,1,2,3,4,5]
	massive = []

	for file in useful_list:
		temp_data = []
		ff = open(str(file)+'.txt','r')
		temp = ff.readlines()
		for i in temp:

			temp_data.append([i[:-2], file])

		massive.extend(temp_data)
	return massive

def word2vec_train(tokenized):
	num_workers = mp.cpu_count()
	if 'ducktale' in tokenized:
		print("111")
	w2vmodel = Word2Vec(tokenized,workers=num_workers,size=num_features, min_count=1)
	w2vmodel.train(tokenized,total_examples=len(tokenized),epochs=50)
	w2vmodel.save('w2vmodel.w2v')

	pass



def main():
	massive = get_data_file()
	tokenized = []
	for i in range(len(massive)):
		current = simple_preprocess(massive[i][0])
		if len(current) > 1:
			tokenized.append(current)
	word2vec_train(tokenized)




if __name__ == '__main__':
    main()

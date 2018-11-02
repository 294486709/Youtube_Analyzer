# Copyright Yixue Zhang jedzhang@bu.edu
import numpy as np
import random
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
FEATURE_NUM = 100


LENGTH_TRAINING = 80
TESTING_PERSENTATGE = 0.10


def get_data_file():
	useful_list = [0,1,2,3,4,5]
	#useful_list = [10, 24, 27, 23, 17, 22, 1, 20, 25, 26, 15]
	massive = []

	for file in useful_list:
		cc = 0
		temp_data = []
		ff = open(str(file)+'.txt','r')
		temp = ff.readlines()
		for i in temp:
			cc += 1
			temp_data.append([i[:-2], file])
			if cc == 2990:
				break
		print(file,len(temp))
		massive.extend(temp_data)

	return massive

def get_w2v(data,model):
	data = simple_preprocess(data)
	result = np.zeros([1,FEATURE_NUM])
	for i in range(len(data)):
		try:
			result = np.vstack((result, model[data[i]]))
		except :
			print(data[i])
			pass
	result = result[1:]
	return result

def main():
	massive = get_data_file()
	massive_mp_x = []
	massive_np_y = []
	useful_list = [0, 1, 2, 3, 4, 5]
	#useful_list = [10, 24, 27, 23, 17, 22, 1, 20, 25, 26, 15]
	model = Word2Vec.load('w2vmodel.w2v')
	print(len(massive))
	c = 0
	for line in massive:
		temp_npy = np.zeros([len(useful_list)])
		current_x = get_w2v(line[0],model)
		current_y = line[1]
		massive_mp_x.append(current_x)
		temp_npy[useful_list.index(current_y)] = 1
		massive_np_y.append(temp_npy)
		c += 1
		print(c/len(massive))
	print(massive_mp_x[len(massive_mp_x)-1])
	combined = list(zip(massive_mp_x,massive_np_y))
	random.shuffle(combined)
		# massive = []
		# for i in range(len(massive_mp_x)):
		# 	massive.append([massive_mp_x[i],massive_np_y[i]])
	massive_mp_x[:], massive_np_y[:] = zip(*combined)
	poplist = []
	for index in range(len(massive_mp_x)):
		if massive_mp_x[index].shape[0] == 0:
			print(index)
			poplist.append(index)
	poplist.sort(reverse=True)
	for i in poplist:
		massive_mp_x.pop(i)
		massive_np_y.pop(i)
	max_length = len(massive_mp_x)
	test_length = int(max_length*TESTING_PERSENTATGE)
	temp = 0
	for i in massive_mp_x:
		if i.shape[0] > temp:
			temp = i.shape[0]
	for i in range(len(massive_mp_x)):
		iterc = 0
		if massive_mp_x[i].shape[0] < 32:
			iterc = temp - massive_mp_x[i].shape[0]
		tttt = np.zeros([1, FEATURE_NUM])
		for k in range(iterc):
			massive_mp_x[i] = np.vstack((massive_mp_x[i],tttt))
		print(massive_mp_x[i].shape[0])
		massive_mp_x[i] = massive_mp_x[i].flatten()
	xtest = massive_mp_x[:test_length]
	ytest = massive_np_y[:test_length]
	xtrain = massive_mp_x[test_length:]
	ytrain = massive_np_y[test_length:]
	np.save('xtrain.npy',xtrain)
	np.save('ytrain.npy',ytrain)
	np.save('xtest',xtest)
	np.save('ytest',ytest)

	return massive_mp_x[test_length:], massive_np_y[test_length:], massive_mp_x[:test_length], massive_np_y[:test_length]






if __name__ == '__main__':
	main()
import presort_v2_v2
import pure_NN
import read_old_data
import word2vec_generator

def main():
	read_old_data.main()
	word2vec_generator.main()
	presort_v2_v2.main()
	pure_NN.main()



if __name__ == '__main__':
    main()
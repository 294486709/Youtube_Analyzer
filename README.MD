Youtube Analyzer(ML part)	

Before you run, make sure you are using Python3.6 and installed all packages that listed below:
tensorflow, numpy, tqdm, gensim.	

please run with "python3 autorun.py".	

The autorun.py will call "read_old_data.py"(fatch data from "USvideos.sorted.csv"), "word2vec_generator.py"(generate w2v dictionary from data), "presort_v2_v2.py"(generate testing and training sets) and "pure_NN.py"(ML Model training) with a given order.	

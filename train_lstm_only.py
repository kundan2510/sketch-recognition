from NN.LSTM import *

from running_average import *
from data_loader_new import *
import os
#from evaluate import *
from keras.utils import generic_utils
from multiprocessing import Process, Queue
from sys import argv


def write_to_file(strng, fname="results_lstm_only_.txt"):
	f = open(fname,'ab')
	f.write(strng+"\n")
	f.close()

def train_single_chunk(model,data,batch_size=5):
	X = data[0]
	Y = data[1]
	load_at_once = len(X)
	print "Got Data"
	avg_loss = 0.0
	avg_acc = 0.0
	pbar = generic_utils.Progbar(load_at_once/batch_size)
	for k in range(load_at_once/batch_size):
		loss,accuracy = model.train_on_batch(X[k*batch_size:(k+1)*batch_size], Y[k*batch_size:(k+1)*batch_size])
		pbar.update(k+1,[("loss",loss),("accuracy",accuracy)])
		avg_loss += loss
		avg_acc += accuracy
	avg_loss = avg_loss/(load_at_once/batch_size)
	avg_acc = avg_acc/(load_at_once/batch_size)
	return avg_loss, avg_acc

def evaluate_single_chunk(model,data,batch_size=5):
	X = data[0]
	Y = data[1]
	load_at_once = len(X)
	_acc = 0.0
	pbar = generic_utils.Progbar(load_at_once/batch_size)
	for k in range(load_at_once/batch_size):
		accuracy = model.evaluate_on_batch(X[k*batch_size:(k+1)*batch_size], Y[k*batch_size:(k+1)*batch_size],with_drop=False)
		_acc += accuracy
		pbar.update(k+1,[("accuracy",accuracy)])
	write_to_file(str(_acc/(load_at_once/batch_size)))

def train(model,X_train,Y_train, X_test, Y_test, num_epochs = 100):
	for i in range(num_epochs):
		avg_loss, avg_acc = train_single_chunk(model,[X_train,Y_train])
		write_to_file("Epoch "+ str(i+1) + " average loss is "+ str(avg_loss) + " average accuracy is "+str(avg_acc))

		if (i + 1)%3 == 0: 
			write_to_file("Calculating test performance after epoch " + str(i))
			evaluate_single_chunk(model,[X_test,Y_test])
			print "Saving model parameters"
			model.save_model_params_dumb("../models/LSTM_ONLY_3_new"+str(i+1)+"_"+str(avg_loss)+"_"+str(avg_acc)+".pkl")

if __name__ == '__main__':
	assert(len(argv) >= 4)
	print "reading arguments"
	train_samples_file = argv[1]
	test_samples_file = argv[2]
	embedding_file = argv[3]
	print "Initializing data loading"
	#loader_train1 = data_loader(train_samples_file, from_chunk = False)
	#loader_train2 = data_loader(train_samples_file, from_chunk = False)
	loader_train = data_loader_new(train_samples_file, embedding_file=embedding_file)
	samples_train_count = loader_train.samples_count
	test_loader = data_loader_new(test_samples_file, embedding_file=embedding_file)
	samples_test_count = test_loader.samples_count

	print "Initializing model"

	model = LSTM(sequence_length = 5)

	print "Loading data"

	X_train,Y_train = loader_train.load_sequence_samples(num_elements = samples_train_count)
	X_test,Y_test = test_loader.load_sequence_samples(num_elements = samples_test_count)

	loader_train = None
	test_loader = None

	train(model,X_train,Y_train,X_test,Y_test,num_epochs=300)




from NN.CNN import *
"""cnn.py contains model architecture"""
from running_average import *
from data_loader_new import *
import os
#from evaluate import *
"""data_loader.py has functions to load samples incrementally such that each batch has class balanced data"""
from keras.preprocessing.image import ImageDataGenerator
from sys import argv
from keras.utils import generic_utils
from multiprocessing import Process, Queue

""" example usage: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -i train.py train_samples.pkl 

	train_ind_sample.pkl is the pickle dump of a dictionary which has
	class index as the key and list of training files as values

"""

def write_to_file(strng, fname="results_new.txt"):
	f = open(fname,'ab')
	f.write(strng+"\n")
	f.close()

def train_single_chunk(model,data,batch_size,load_at_once):
	X = data[0]
	Y = data[1]
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


def evaluate_single_chunk_par(model,data,batch_size,load_at_once,test_datagen):
        X = data[0]
        Y = data[1]
	load_at_once = len(X)
	_acc = 0.0
	pbar = generic_utils.Progbar(load_at_once/batch_size)
        for k in range(load_at_once/batch_size):
                accuracy = model.evaluate_on_batch(X[k*batch_size:(k+1)*batch_size], Y[k*batch_size:(k+1)*batch_size],with_drop=False)
                _acc += accuracy
                pbar.update(k+1,[("accuracy",accuracy)])
	rem_eval = load_at_once - (load_at_once/batch_size)*batch_size
	if rem_eval > 0:
		X_batch = X[-rem_eval::]
		Y_batch = Y[-rem_eval::]
		accuracy = model.evaluate_on_batch(X_batch, Y_batch,with_drop=False)
		_acc += accuracy*batch_size/rem_eval
		_acc = _acc/((load_at_once/batch_size)+1)
	else:
		_acc = _acc/(load_at_once/batch_size)
		
        write_to_file("Test performance is "+ str(_acc))



def train_model_with_parallel_loading(model,q_train, X_test,Y_test, batch_size=8,load_at_once=1000, num_epoch=100, samples_train_count=13000,samples_test_count=7000, datagen = None,test_data_gen=None):
	#load_at_once = len(X)
	num_batch_per_epoch = int((samples_train_count*1.0/load_at_once))
	for i in range(num_epoch):
		avg_loss = 0.0
		avg_acc = 0.0
		for j in range(num_batch_per_epoch):
			print("\nEpoch no. %s, batch_no. %s\n"%(i+1,j+1))
			data = q_train.get()
			loss, acc = train_single_chunk(model,data,batch_size,load_at_once)
			avg_loss += loss
			avg_acc += acc
		avg_acc = avg_acc/num_batch_per_epoch
		avg_loss = avg_loss/num_batch_per_epoch
		write_to_file("Epoch "+ str(i) + " average loss is "+ str(avg_loss) + " average accuracy is "+str(avg_acc))
		if (i + 1)%3 == 0: 
			write_to_file("Calculating test performance after epoch " + str(i))
			evaluate_model_with_parallel_loading(model,X_test,Y_test,batch_size=batch_size,num_epoch=1,samples_test_count=samples_test_count,test_data_gen=test_data_gen)

		if((i+1) % 3 == 0):
			print "Saving the model in epoch %s"%(str(i+1))
			model.save_model_params_dumb("/home/kundan/models/CNN_no_shear_new_3drop_"+str(i+1)+"_"+str(avg_loss)+"_"+str(avg_acc)+".pkl")
	return model


def evaluate_model_with_parallel_loading(model,X_test,Y_test,batch_size=8,load_at_once=200, num_epoch=1,samples_test_count=7000,test_data_gen=None):
        load_at_once = len(X_test)
        num_batch_per_epoch = int((samples_test_count*1.0/load_at_once))
        for i in range(num_epoch):
                for j in range(num_batch_per_epoch):
                        print("\nEpoch no. %s, batch_no. %s\n"%(i+1,j+1))
                        data = [X_test,Y_test]
                        evaluate_single_chunk_par(model,data,batch_size,load_at_once,test_data_gen)
	return model

if __name__ == '__main__':
	assert(len(argv) >= 2)
	print "reading arguments"
	train_samples_file = argv[1]
	test_samples_file = argv[2]
	print "Initializing data loading"
	#loader_train1 = data_loader(train_samples_file, from_chunk = False)
	#loader_train2 = data_loader(train_samples_file, from_chunk = False)
	loader_train = data_loader_new(train_samples_file)
	samples_train_count = loader_train.samples_count
	test_loader = data_loader_new(test_samples_file)
	samples_test_count = test_loader.samples_count
	q_train = Queue()
	# q_test = Queue()

	if os.fork() == 0:
		i = 0
	 	while True:
	 		if q_train.qsize() < 30:
	 			i = (i + 1)%97
	 			if i % 20 != 0:
		 			X , Y = loader_train.load_samples(num_elements=1008,transform=True)
		 			#print "Loaded batch"
		 			q_train.put([X,Y])
		 		else:
		 			X , Y = loader_train.load_samples(num_elements=1008,transform=False)
		 			#print "Loaded batch"
		 			q_train.put([X,Y])
	# if os.fork() == 0:
	# 	while True:
	# 		if q_test.qsize() < 5:
	# 			Xt , Yt = test_loader.load_samples(num_elements=250)
	# 			#print "Loaded batch"
	# 			q_test.put([Xt,Yt])
	#X,Y = loader_train.load_samples(num_elements = samples_train_count)
	X_test, Y_test = test_loader.load_samples(num_elements=samples_test_count,transform=False)
	
	#datagen = ImageDataGenerator(featurewise_std_normalization=True,featurewise_center=True,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2, horizontal_flip=True)
	#datagen.fit(X)
	#test_data_gen = ImageDataGenerator(featurewise_std_normalization=True,featurewise_center=True)
	#test_data_gen.fit(X)
	print "Creating CNN architecture....."
	model = CNN(batch_size=16)
	# model_arch_json = model.to_json()
	# pickle.dump(model_arch_json,open('model_cnn_more_droput.json.pkl','wb'))
	print "CNN architechture created"
	print "Starting Training..."
	#num_evaluate = 10
	#for i in range(num_evaluate):
	#	model = train_model_with_parallel_loading(model,loader,num_epoch=2)
	#	write_to_file("Evaluating model performance\n")
	#	model = evaluate_model_with_parallel_loading(model,test_loader,num_epoch=1)
	#model = train_model(model,loader)

	model = train_model_with_parallel_loading(model,q_train, X_test, Y_test, batch_size = 16, num_epoch=1000, samples_train_count=samples_train_count, samples_test_count=samples_test_count, datagen=None, test_data_gen=None)



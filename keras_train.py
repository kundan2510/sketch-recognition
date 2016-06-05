from cnn import *
"""cnn.py contains model architecture"""
from running_average import *
from data_loader import *
from evaluate import *
"""data_loader.py has functions to load samples incrementally such that each batch has class balanced data"""

from sys import argv
from keras.utils import generic_utils
from multiprocessing import Process, Queue

""" example usage: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -i train.py train_samples.pkl 

	train_ind_sample.pkl is the pickle dump of a dictionary which has
	class index as the key and list of training files as values

"""
def write_to_file(strng, fname="results_keras.txt"):
	f = open(fname,'ab')
	f.write(strng+"\n")
	f.close()

def train_single_chunk(model,q,batch_size,load_at_once):
	data = q.get()
	X = data[0]
	Y = data[1]
	print "Got Data"
	avg_loss = 0.0
	avg_acc = 0.0
	pbar = generic_utils.Progbar(load_at_once/batch_size)
	for k in range(load_at_once/batch_size):
		loss,accuracy = model.train_on_batch(X[k*batch_size:(k+1)*batch_size], Y[k*batch_size:(k+1)*batch_size],accuracy= True)
		pbar.update(k+1,[("loss",loss),("accuracy",accuracy)])
		avg_loss += loss
		avg_acc += accuracy
	avg_loss = avg_loss/(load_at_once/batch_size)
	avg_acc = avg_acc/(load_at_once/batch_size)
	return avg_loss, avg_acc


def evaluate_single_chunk_par(model,q,batch_size,load_at_once):
        data = q.get()
        X = data[0]
        Y = data[1]
        _acc = 0.0
        #print "Got Data"
        pbar = generic_utils.Progbar(load_at_once/batch_size)
        for k in range(load_at_once/batch_size):
                loss, accuracy = model.test_on_batch(X[k*batch_size:(k+1)*batch_size], Y[k*batch_size:(k+1)*batch_size],accuracy=True)
                _acc += accuracy
		pbar.update(k+1,[("accuracy",accuracy),("loss",loss)])
	write_to_file(str(_acc/(load_at_once/batch_size)))

def load_single_in_queue(q,loader,load_at_once):
	#print ("Loading next batch of data")
	X , Y = loader.load_samples(num_elements=load_at_once)
	#print "Loaded batch"
	q.put([X,Y])
	return

def train_model_with_parallel_loading(model,loader,batch_size=8,load_at_once=400, num_epoch=100,test_loader=None):
	num_batch_per_epoch = int((loader.samples_count/load_at_once))
	q = Queue()
	X_old , Y_old = loader.load_samples(num_elements=load_at_once)
	q.put([X_old,Y_old])
	for i in range(num_epoch):
		avg_loss = 0.0
		avg_acc = 0.0
		for j in range(num_batch_per_epoch):
			print("\nEpoch no. %s, batch_no. %s\n"%(i+1,j+1))
			p1 = Process(target=load_single_in_queue,args=(q,loader,load_at_once))
			p1.start()
			loss, acc = train_single_chunk(model,q,batch_size,load_at_once)
			avg_loss += loss
			avg_acc += acc
		avg_acc = avg_acc/num_batch_per_epoch
		avg_loss = avg_loss/num_batch_per_epoch

		if test_loader:
			write_to_file("Done with epoch %d\n"%(i+1))
			evaluate_model_with_parallel_loading(model,test_loader,num_epoch=1)
		
		#if((i+1) % 4 == 0):
		#	print "Saving the model in epoch %s"%(str(i+1))
		#	model.save_model_params_dumb("CNN_"+str(i+1)+"_"+str(avg_loss)+"_"+str(avg_acc)+".pkl")
	return model


def evaluate_model_with_parallel_loading(model,loader,batch_size=8,load_at_once=400, num_epoch=1):
        num_batch_per_epoch = int((loader.samples_count/load_at_once))
        q_test = Queue()
        X_old , Y_old = loader.load_samples(num_elements=load_at_once)
        q_test.put([X_old,Y_old])
        for i in range(num_epoch):
                for j in range(num_batch_per_epoch):
                        print("\nEpoch no. %s, batch_no. %s\n"%(i+1,j+1))
                        p1 = Process(target=load_single_in_queue,args=(q_test,loader,load_at_once))
                        p1.start()
                        evaluate_single_chunk_par(model,q_test,batch_size,load_at_once)
	return model

def train_model(model,loader,batch_size=4, load_at_once=250, num_epoch=100):
	num_batch_per_epcoh = int((loader.samples_count/load_at_once))
	for i in range(num_epoch):
		for j in range(num_batch_per_epcoh):
			print("\nEpoch no. %s, loading batch no. %s\n"%(i+1,j+1))
			X,Y = loader.load_samples(num_elements=load_at_once)
			print("Training......")
			pbar = generic_utils.Progbar(load_at_once/batch_size)
			for k in range(load_at_once/batch_size):
				loss,accuracy = model.train_on_batch(X[k*batch_size:(k+1)*batch_size], Y[k*batch_size:(k+1)*batch_size])
				pbar.update(k+1,[("loss",loss),("accuracy",accuracy)])
	return model

def train_model_chunks(model,loader,batch_size=8,num_epoch=100,print_every=10,epoch_done=0):

	loss_avg = running_average(100)
	acc_avg = running_average(100)

	for i in range(num_epoch):
		for j in range(loader.num_chunks):
			print("\nEpoch no. %s, loading chunk no. %s\n"%(epoch_done+i+1,j+1))
			X,Y = loader.next_chunk()
			for k in range(len(Y)/batch_size):
				loss, accuracy = model.train_on_batch(X[k*batch_size:(k+1)*batch_size], Y[k*batch_size:(k+1)*batch_size])
				curr_avg_loss = loss_avg.upsert(loss)
				curr_avg_acc = acc_avg.upsert(accuracy)
				if (k+1)%print_every == 0:
					print("Epoch: %d, chunk: %d, batch: %d, loss=%.4f, accuracy=%.4f"%(epoch_done+i+1,j+1,k+1,curr_avg_loss,curr_avg_acc))
		# if (i+1) % 5 == 0:
		# 	model.save_weights('model_weights_cnn_'+str(epoch_done+i+1)+'_'+str(curr_avg_acc)+'.h5',overwrite=True)
	return model

if __name__ == '__main__':
	assert(len(argv) >= 2)
	print "reading arguments"
	train_samples_file = argv[1]
	test_samples_file = argv[2]
	print "Initializing data loading"
	loader = data_loader(train_samples_file, from_chunk = False)
	test_loader = data_loader(test_samples_file, from_chunk = False)
	print "Creating CNN architecture....."
	#model = CNN(batch_size=8)
	model = create_model()
	# model_arch_json = model.to_json()
	# pickle.dump(model_arch_json,open('model_cnn_more_droput.json.pkl','wb'))
	print "CNN architechture created"
	print "Starting Training..."
	num_evaluate = 10
	#for i in range(num_evaluate):
	#	model = train_model_with_parallel_loading(model,loader,num_epoch=2)
	#	write_to_file("Evaluating model performance\n")
	#	model = evaluate_model_with_parallel_loading(model,test_loader,num_epoch=1)
	#model = train_model(model,loader)

	model = train_model_with_parallel_loading(model,loader,num_epoch=300, test_loader=test_loader)



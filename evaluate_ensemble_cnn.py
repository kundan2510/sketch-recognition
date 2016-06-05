import theano
from train import *

def check_batch(Y_truth, Y_predictions):
	Y_true = np.argmax(Y_truth,axis=1)
	Y_new_predictions = np.argmax(np.sum(Y_predictions, axis=0),axis = 1)
	#Y_new_predictions = np.argmax(Y_predictions[0],axis = 1)
	#print len(Y_true)
	#print np.shape(Y_predictions)
	Y_final_predictions = Y_new_predictions[0:len(Y_true)]
	#print "Y_true\n",Y_true
	#print "Y_predictions\n",Y_final_predictions
	#print np.sum(Y_true==Y_final_predictions)
	return np.sum(Y_true==Y_final_predictions)

def evaluate_ensemble(CNN_list,X_test,Y_test, model_importances, batch_size):
	num_data_points = len(X_test)
	run_loop = num_data_points/batch_size;
	if (num_data_points % batch_size) != 0:
		run_loop += 1
	accurate = 0
	pbar = generic_utils.Progbar(run_loop)
	for k in range(run_loop):
		prediction_on_batch = np.zeros((len(CNN_list),batch_size,250))
		for i in range(len(CNN_list)):
			prediction_on_batch[i,0:len(X_test[k*batch_size:(k+1)*batch_size]),:] = model_importances[i]*CNN_list[i].predict_prob_on_batch(X_test[k*batch_size:(k+1)*batch_size])
		accurate += check_batch(Y_test[k*batch_size: (k+1)*batch_size], prediction_on_batch)
		#print accurate
		pbar.update(k+1,[("accuracy",accurate*1.0/min((k+1)*batch_size,num_data_points))])
	print "accuracy is %.4f"%(accurate*1.0/num_data_points)

def find_accuracy(Y_truth,Y_single_cnn_prediction):
	Y_true = np.argmax(Y_truth, axis=1)
	Y_pred = np.argmax(Y_single_cnn_prediction,axis=1)
	return np.sum(Y_true==Y_pred)
		
if __name__ == "__main__":
	validation_file = argv[1]
	CNN_weights_files = argv[2::]
	CNN_list = [CNN(batch_size=16) for i in range(len(CNN_weights_files))]
	for i,f in enumerate(CNN_weights_files):
		CNN_list[i].load_model_params_dumb(f)

	test_loader = data_loader_new(validation_file)
        samples_test_count = test_loader.samples_count
	X_test, Y_test = test_loader.load_samples(num_elements=samples_test_count,transform=False)
	test_loader = None
	
	model_importances = [0.63,0.34,0.70]
	# evaluate_single_chunk_par(CNN_list[0],[X_test,Y_test],16,0,None)
	# evaluate_single_chunk_par(CNN_list[1],[X_test,Y_test],16,0,None)
	# evaluate_single_chunk_par(CNN_list[2],[X_test,Y_test],16,0,None)
	evaluate_ensemble(CNN_list,X_test, Y_test, model_importances, 16)

	
	

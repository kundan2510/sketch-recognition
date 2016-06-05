from data_loader import *
from keras.models import model_from_json
import cPickle as pickle
from sys import argv
from train import *

if __name__ == '__main__':
	assert(len(argv) >= 3)
	train_samples = argv[1]
	model_weights = argv[2]
	train_loader = data_loader(train_samples,from_chunk=False,transform=False)
	print "Loading model configs..."
	#model_json = pickle.load(open(argv[2],'rb'))
	#model = model_from_json(model_json)
	model = CNN(batch_size=8)
	print "Loading model weights..."
	#model.load_weights(argv[3])
	model.load_model_params_dumb(model_weights)
	model = train_model_with_parallel_loading(model,train_loader,num_epoch=16)
	#test_using_chunks(model,test_data_loader)

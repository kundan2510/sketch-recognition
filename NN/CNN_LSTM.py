from layers import *
from costfunctions import *
from theano import Param
import numpy as np
from lib import *
import cPickle
import gzip

class LSTM_CNN:
	def __init__(self, sequence_length = 5):
		self.sequence_length = sequence_length
		self.num_updates = 0
		layers = []
		X = T.matrix().reshape((sequence_length,1,225,225))

		Y = T.ivector()

		is_train = T.scalar()

		inputs = InputLayer(X,name="input")
		layers.append(inputs)

		convlayer0 = ConvLayer(inputs,(64,1,15,15), subsample=(3,3) ,name="conv 0")
		relu0 = RELU(convlayer0, name="relu 0")
		pool0 = PoolLayer(relu0, pool_size=(3,3), stride=(2,2), name="pool 0")

		layers += [convlayer0,relu0,pool0]

		convlayer1 = ConvLayer(pool0,(128,64,5,5), name="conv 1")
		relu1 = RELU(convlayer1, name="relu 1")
		pool1 = PoolLayer(relu1, pool_size=(3,3), stride=(2,2), name="pool 1")

		layers += [convlayer1,relu1,pool1]

		convlayer2 = ConvLayer(pool1,(256,128,3,3),border_mode=(1,1), name="conv 2")
		relu2 = RELU(convlayer2, name="relu 2")

		layers += [convlayer2,relu2]

		convlayer3 = ConvLayer(relu2,(256,256,3,3),border_mode=(1,1) , name="conv 3")
		relu3 = RELU(convlayer3, name="relu 3")

		layers += [convlayer3,relu3]

		convlayer4 = ConvLayer(relu3,(256,256,3,3),border_mode=(1,1), name="conv 4")
		relu4 = RELU(convlayer4, name="relu 4")
		pool2 = PoolLayer(relu4, pool_size=(3,3), stride=(2,2), name="pool 2")


		layers += [convlayer4,relu4,pool2]

		convlayer5 = ConvLayer(pool2,(512,256,7,7),name="conv 5")
		relu5 = RELU(convlayer5,name="relu 5")

		squeezed = Squeeze(relu5,outdim=2, name="squeeze 1")

		drop1 = DropoutLayer(squeezed, is_train, p = 0.5, name="Drop 1, p = 0.5")


		layers += [convlayer5, relu5, squeezed, drop1]

		lstm1 = LSTMLayer(drop1,512,512,name="LSTM 1",return_sequences=True)
		# layers += [convlayer5, relu5, squeezed]

		# fullyconn1 = FCLayer(squeezed,512,512,name="FC 1")
		drop2 = DropoutLayer(lstm1,is_train, p = 0.5, name="Drop 2, p = 0.5")


		# fullyconn2 = FCLayer(fullyconn1,512,250,name="FC 2")
		lstm2 = LSTMLayer(drop2,512,250,name="lstm 2",return_sequences=False)
		
		softmax1 = SoftmaxLayer(lstm2, name="softmax")


		# layers += [fullyconn1,fullyconn2,softmax1]
		layers += [lstm1,lstm2,drop2,softmax1]
		
		predicted_class = T.argmax(softmax1.output())

		cost = CategoricalCrossEntropy(softmax1,Y).output()


		self.layers = layers

		params = get_params(self.layers)
		biases = get_biases(self.layers)

		caches_params = make_caches(params)
		caches_bias = make_caches(biases)
		eta = T.scalar()

		updates = momentum(cost, params, biases, caches_params,caches_bias, eta)
		
		self.train = theano.function([X,Y,eta, Param(is_train,1)],[cost,predicted_class],updates=updates,allow_input_downcast=True,on_unused_input='warn')

		self.predict = theano.function([X,Param(is_train,0)],predicted_class,allow_input_downcast=True,on_unused_input='warn')

		self.predict_with_drop = theano.function([X,Param(is_train,1)],predicted_class,allow_input_downcast=True, on_unused_input='warn')	

		self.validate = theano.function([X,Y,Param(is_train,0)],cost,allow_input_downcast=True, on_unused_input='warn')

	def find_accuracy(self,Y, predicted_class):
		return np.mean(np.argmax(Y,axis=1) == np.asarray(predicted_class))

	def train_on_batch(self, X, Y, eta=0.01,eta_decay_rate= 5e-5):
		eta = eta/(1 + self.num_updates*eta_decay_rate)
		self.num_updates += 1
		train_labels = np.argmax(Y,axis=1)
		# predicted_class_using_predict_function = self.predict(X,1)
		cost = 0.0; predicted_class = []
		for i in range(len(X)):
			_cost, _predicted_class = self.train(X[i],train_labels[i:i+1],eta,1)
			cost += _cost
			predicted_class += [_predicted_class]
		
		# print np.argmax(Y,axis=1)
		# print np.asarray(predicted_class)
		# print (np.argmax(Y,axis=1) == np.asarray(predicted_class))
		# print np.mean(np.argmax(Y,axis=1) == np.asarray(predicted_class))
		accuracy = self.find_accuracy(Y,predicted_class)
		return cost/len(X), accuracy

	def predict_on_batch(self,X,with_drop=False):
		Y = []
		if with_drop:
			for i in range(len(X)):
				Y += [self.predict_with_drop(X[i],1)]
		else:
			for i in range(len(X)):
				Y += [self.predict(X[i],0)]
		#print Y
		return Y

	def evaluate_on_batch(self,X,Y,with_drop=False):
		predictions = self.predict_on_batch(X,with_drop)
		# print np.argmax(Y,axis=1)
		# print np.asarray(predictions)
		accuracy = self.find_accuracy(Y, predictions)
		return accuracy


	def save_model_params_dumb(self, filename):
        	to_save = { 'sequence_length': self.sequence_length, 'num_updates': self.num_updates, 'layers_name':"", 'layers_num' :len(self.layers)}
	 
		for l in self.layers:
			to_save['layers_name'] += l.name
			for p in l.get_params() + l.get_bias():
				assert(p.name not in to_save)
				to_save[p.name] = p.get_value()

		with gzip.open(filename, 'wb') as f:
			cPickle.dump(to_save, f)

	
	def load_model_params_dumb(self, filename):
		f = gzip.open(filename, 'rb')
		to_load = cPickle.load(f)
		assert(to_load['sequence_length'] == self.sequence_length)
		assert(to_load['layers_num'] == len(self.layers))
		for l in self.layers:
			assert(l.name in to_load['layers_name'])
			for p in l.get_params()+l.get_bias():
				p.set_value(floatX(to_load[p.name]))
	

	def load_pre_trained_cnn(self, filename):
		f = gzip.open(filename, 'rb')
		to_load = cPickle.load(f)
		for l in self.layers:
			if l.name in to_load['layers_name']:
				for p in l.get_params()+l.get_bias():
					p.set_value(floatX(to_load[p.name]))
	

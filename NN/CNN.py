from layers import *
from costfunctions import *
from theano import Param
import numpy as np
from lib import *
import cPickle
import gzip

class CNN:
	def __init__(self, batch_size= 8):
		self.batch_size = batch_size
		self.num_updates = 0
		layers = []
		X = T.matrix().reshape((batch_size,1,225,225))

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


		drop0 = DropoutLayer(pool2, is_train, p = 0.5, name="Drop 0, p = 0.5")
		layers += [convlayer4,relu4,pool2,drop0]

		convlayer5 = ConvLayer(drop0,(512,256,7,7),name="conv 5")
		relu5 = RELU(convlayer5,name="relu 5")

		squeezed = Squeeze(relu5,outdim=2, name="squeeze 1")


		drop1 = DropoutLayer(squeezed, is_train, p = 0.5, name="Drop 1, p = 0.5")

		layers += [convlayer5, relu5, squeezed, drop1]
		# layers += [convlayer5, relu5, squeezed]

		fullyconn1 = FCLayer(drop1,512,512,name="FC 1")
		# fullyconn1 = FCLayer(squeezed,512,512,name="FC 1")

		relu6 = RELU(fullyconn1,name="relu 6")

		drop2 = DropoutLayer(relu6, is_train, p = 0.5, name="Drop 2, p = 0.5")


		#fullyconn2 = FCLayer(fullyconn1,512,250,name="FC 2")
		fullyconn2 = FCLayer(drop2,512,250,name="FC 2")
		
		softmax1 = SoftmaxLayer(fullyconn2, name="softmax")


		# layers += [fullyconn1,fullyconn2,softmax1]
		layers += [fullyconn1,drop2, relu6, fullyconn2,softmax1]
		
		predicted_class = T.argmax(softmax1.output(), axis=1)

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
		
		self.predict_prob = theano.function([X,Param(is_train,0)],softmax1.output(), allow_input_downcast=True,on_unused_input='warn')
		
		self.validate = theano.function([X,Y,Param(is_train,0)],cost,allow_input_downcast=True, on_unused_input='warn')

		self.get_embeddings = theano.function([X,Param(is_train,0)],fullyconn1.output(), allow_input_downcast=True, on_unused_input='warn')

	def find_accuracy(self,Y, predicted_class):
		return np.mean(np.argmax(Y,axis=1) == predicted_class)

	def train_on_batch(self, X, Y, eta=0.01,eta_decay_rate= 5e-5):
		#assert(len(X) == self.batch_size)
		eta = eta/(1 + self.num_updates*eta_decay_rate)
		self.num_updates += 1

		# predicted_class_using_predict_function = self.predict(X,1)
		if self.num_updates % 29 == 0:
			cost, predicted_class = self.train(X,np.argmax(Y,axis=1),eta,0)
		else:
			cost, predicted_class = self.train(X,np.argmax(Y,axis=1),eta,1)
		# if np.sum(predicted_class == predicted_class_using_predict_function) != len(predicted_class):
  #                       print "Problem is here ",predicted_class, predicted_class_using_predict_function

		
		accuracy = self.find_accuracy(Y,predicted_class)
		return cost, accuracy

	def predict_on_batch(self,X,with_drop=False):
		#assert(len(X) == self.batch_size)
		if with_drop:
			Y = self.predict_with_drop(X,1)
		else:
			Y = self.predict(X,0)
		#print Y
		return Y

	def predict_prob_on_batch(self,X,with_drop=False):
		if with_drop:
			Y = self.predict_prob(X,1)
		else:
			Y = self.predict_prob(X,0)

		return Y

	def evaluate_on_batch(self,X,Y,with_drop=False):
		predictions = self.predict_on_batch(X,with_drop)
		#print ground_truth
		#print predictions	
		accuracy = self.find_accuracy(Y, predictions)
		return accuracy
	
	def bayesian_prediction(self,X):
		Y_error = np.zeros(250)
		Y_pred = np.ones(len(X))*-1
		for i in range(len(X)):
			X_new = np.tile(X[i],(self.batch_size,1,1,1))
			for j in range(250):
				Y_error[j] = self.validate(X_new,[j]*self.batch_size)
			Y_pred[i] = np.argmin(Y_error)
		return Y
			
	def save_model_params_dumb(self, filename):
        	to_save = { 'batch_size': self.batch_size, 'num_updates': self.num_updates, 'layers_name':"", 'layers_num' :len(self.layers)}
	 
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
		self.num_updates = to_load['num_updates']
		#assert(to_load['batch_size'] == self.batch_size)
		#assert(to_load['layers_num'] == len(self.layers))
		for l in self.layers:
			assert(l.name in to_load['layers_name'])
			for p in l.get_params()+l.get_bias():
				p.set_value(floatX(to_load[p.name]))
	

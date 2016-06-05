from layers import *
import numpy as np
from costfunctions import *

def convlayer_check():
	X = T.matrix().reshape((5,3,256,256))
	inputs = InputLayer(X, name="inputs")
	convlayer = ConvPoolLayer(inputs,(5,3,10,10))
	print convlayer.get_weight()
	print np.shape(convlayer.get_weight())
	out = theano.function([X],convlayer.output())
	img = np.ones((5,3,256,256))
	return out(img)

def test_theano():
	a = T.scalar()
	b = T.scalar()
	c = a + b;
	out = theano.function([a,b],c)
	print out(3,4)

def create_model(batch_size):
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

	convlayer2 = ConvLayer(pool1,(256,128,3,3), name="conv 2")
	relu2 = RELU(convlayer2, name="relu 2")

	layers += [convlayer2,relu2]

	convlayer3 = ConvLayer(relu2,(256,256,3,3), name="conv 3")
	relu3 = RELU(convlayer3, name="relu 3")

	layers += [convlayer3,relu3]

	convlayer4 = ConvLayer(relu3,(256,256,3,3), name="conv 4")
	relu4 = RELU(convlayer4, name="relu 4")
	pool2 = PoolLayer(relu4, pool_size=(3,3), stride=(2,2), name="pool 2")

	layers += [convlayer4,relu4,pool2]

	convlayer5 = ConvLayer(pool2,(512,256,4,4),name="conv 5")
	relu5 = RELU(convlayer5,name="relu 5")

	squeezed = Squeeze(relu5,outdim=2, name="squeeze 1")

	drop1 = DropoutLayer(squeezed, is_train, p = 0.5, name="Drop 1, p = 0.5")

	layers += [convlayer5, relu5, squeezed, drop1]

	fullyconn1 = FCLayer(drop1,512,512,name="FC 1")

	drop2 = DropoutLayer(fullyconn1,is_train, p = 0.5, name="Drop 2, p = 0.5")

	fullyconn2 = FCLayer(drop2,512,250,name="FC 2")

	softmax1 = SoftmaxLayer(fullyconn2, name="softmax")

	layers += [fullyconn1,drop2,fullyconn2,softmax1]
	
	cost = CategoricalCrossEntropy(softmax1,Y).output()

	train = theano.function([X,is_train,Y],cost)
		
	return test


def check_output_shape():
	model = create_model(2)
	img = np.ones((2,1,225,225))
	print "Shape of output is ",(np.shape(model(img,1,[1,3])))

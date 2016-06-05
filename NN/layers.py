import numpy as np
import theano
import theano.tensor as T
from lib import sigmoid, softmax, dropout, floatX, random_weights, zeros

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from theano.sandbox.cuda.dnn import dnn_conv

class NNLayer:
	def get_params(self):
		return self.params

	def save_model(self):
		return

	def load_model(self):
		return

	def updates(self):
		return []

	def reset_state(self):
		return

	def get_bias(self):
		return self.bias


class LSTMLayer(NNLayer):

    def __init__(self, input_layer, num_input, num_cells,batch_size = 8, name="", go_backwards=False, return_sequences = False):
        """
        LSTM Layer
        """

        self.name = name
        self.num_input = num_input
        self.num_cells = num_cells

        self.return_sequences = return_sequences

        self.X = input_layer.output()

        self.h0 = theano.shared(floatX(np.zeros(num_cells,)))
        self.s0 = theano.shared(floatX(np.zeros(num_cells,)))

        self.go_backwards = go_backwards

        W_bound_sx = np.sqrt(6. / (num_input + num_cells))
        rng = np.random.RandomState(23456)

        self.W_gx = theano.shared(np.asarray(rng.uniform(low=-W_bound_sx, high=W_bound_sx, size=(num_input,num_cells)), dtype=theano.config.floatX), name = name + " W_gx",borrow=True)
        self.W_ix = theano.shared(np.asarray(rng.uniform(low=-W_bound_sx, high=W_bound_sx, size=(num_input,num_cells)), dtype=theano.config.floatX), name = name + " W_ix",borrow=True)
        self.W_fx = theano.shared(np.asarray(rng.uniform(low=-W_bound_sx, high=W_bound_sx, size=(num_input,num_cells)), dtype=theano.config.floatX), name = name + " W_fx",borrow=True)
        self.W_ox = theano.shared(np.asarray(rng.uniform(low=-W_bound_sx, high=W_bound_sx, size=(num_input,num_cells)), dtype=theano.config.floatX), name = name + " W_ox",borrow=True)

        W_bound_sh = np.sqrt(6. / (num_cells + num_cells))

        self.W_gh = theano.shared(np.asarray(rng.uniform(low=-W_bound_sh, high=W_bound_sh, size=(num_cells,num_cells)), dtype=theano.config.floatX), name = name + " W_gh",borrow=True)
        self.W_ih = theano.shared(np.asarray(rng.uniform(low=-W_bound_sh, high=W_bound_sh, size=(num_cells,num_cells)), dtype=theano.config.floatX), name = name + " W_ih",borrow=True)
        self.W_fh = theano.shared(np.asarray(rng.uniform(low=-W_bound_sh, high=W_bound_sh, size=(num_cells,num_cells)), dtype=theano.config.floatX), name = name + " W_fh",borrow=True)
        self.W_oh = theano.shared(np.asarray(rng.uniform(low=-W_bound_sh, high=W_bound_sh, size=(num_cells,num_cells)), dtype=theano.config.floatX), name = name + " W_oh",borrow=True)

        self.b_g = random_weights((num_cells,), name=self.name+" b_g")
        self.b_i = random_weights((num_cells,), name=self.name+" b_i")
        self.b_f = random_weights((num_cells,), name=self.name+" b_f")
        self.b_o = random_weights((num_cells,), name=self.name+" b_o")

        self.params = [self.W_gx, self.W_ix, self.W_ox, self.W_fx,
                        self.W_gh, self.W_ih, self.W_oh, self.W_fh,
                ]
	self.bias = [self.b_g, self.b_i, self.b_f, self.b_o]

    def one_step(self, x, h_tm1, s_tm1):
        """
        """
        g = T.tanh(T.dot(x, self.W_gx) + T.dot(h_tm1, self.W_gh) + self.b_g)
        i = T.nnet.sigmoid(T.dot(x, self.W_ix) + T.dot(h_tm1, self.W_ih) + self.b_i)
        f = T.nnet.sigmoid(T.dot(x, self.W_fx) + T.dot(h_tm1, self.W_fh) + self.b_f)
        o = T.nnet.sigmoid(T.dot(x, self.W_ox) + T.dot(h_tm1, self.W_oh) + self.b_o)

        s = i*g + s_tm1 * f
        h = T.tanh(s) * o

        return h, s


    def output(self):

        outputs_info = [self.h0, self.s0]

        ([outputs, states], updates) = theano.scan(
                fn=self.one_step,
                sequences=[self.X],
                outputs_info = outputs_info
            )
        if self.return_sequences:
            return outputs
        else:
            return outputs[-1]


    # def updates(self):
    #     return [(self.s0, self.new_s), (self.h0, self.new_h)]

    def reset_state(self):
        self.h0 = theano.shared(floatX(np.zeros(self.num_cells)))
        self.s0 = theano.shared(floatX(np.zeros(self.num_cells)))

#Class to make any layer recurrent to run on arbitrary length sequences, this is a kind of map function which takes a list of input and layer, and gives a list of layers output for each input
class RecurrentLayer(NNLayer):
	def __init__(self,layer):
		raise "Not Implemented Error"


class FCLayer(NNLayer):
    """
    """
    def __init__(self, input_layer, num_input, num_output, name=""):
    	self.X = input_layer.output()
	self.name = name
        W_bound = np.sqrt(6. / (num_input + num_output))
        rng = np.random.RandomState(23495)
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(num_input,num_output)), dtype=theano.config.floatX), name = name + " W",borrow=True)
	self.b = random_weights((num_output,), name=name+"_b")
	self.params = [self.W]
	self.bias = [self.b]

    def output(self):
        return T.dot(self.X, self.W) + self.b


class InputLayer(NNLayer):
    """
    """
    def __init__(self, X, name=""):
        self.name = name
        self.X = X
        self.params=[]
	self.bias = []

    def output(self):
        return self.X


class SoftmaxLayer(NNLayer):
    """
    """
    def __init__(self, input_layer, name=""):
        self.name = name
        self.X = input_layer.output()
        self.params = []
	self.bias = []

    def output(self):
        return T.nnet.softmax(self.X)

class ConvLayer(NNLayer):
    def __init__(self, input_layer, filter_shape, subsample=(1,1),name="",border_mode='valid'):
        self.X = input_layer.output()
	self.border_mode = border_mode
        self.name = name
        self.subsample = subsample
        self.filter_shape = filter_shape
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(subsample))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        rng = np.random.RandomState(23455)	
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX), name = name + " W",borrow=True)
        self.b = random_weights((filter_shape[0],), name=name+" bias")
        self.params = [self.W]
	self.bias = [self.b]

        self.get_weight = theano.function([],self.W)

    def output(self):
        # conv_out = conv.conv2d( input=self.X, filters=self.W, filter_shape=self.filter_shape, subsample=self.subsample)
        conv_out = dnn_conv( self.X, self.W, border_mode = self.border_mode, subsample=self.subsample)
        return conv_out + self.b.dimshuffle('x',0,'x','x')


class PoolLayer(NNLayer):
	def __init__(self,input_layer,pool_size=(2,2), stride = None, name=""):
		self.X = input_layer.output()
		self.name = name
		self.params = []
		self.bias = []
		self.pool_size = pool_size

		if not stride:
			stride = pool_size
		self.stride = stride 
	
	def output(self):
		return downsample.max_pool_2d(input=self.X, ds=self.pool_size, st= self.stride, ignore_border=True)

class RELU(NNLayer):
	def __init__(self,input_layer,name=""):
		self.X = input_layer.output()
		self.name = name
		self.params = []
		self.bias = []
	
	def output(self):
		return T.nnet.relu(self.X)


class SigmoidLayer(NNLayer):

    def __init__(self, num_input, num_output, input_layers, name=""):
	self.name = name
        if len(input_layers) >= 2:
            print "number of input layers: %s" % len(input_layers)
            print "len of list comprehension: %s" % len([input_layer.output() for input_layer in input_layers])
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()
        self.W_yh = random_weights((num_input, num_output),name=name+"_W_yh")
        self.b_y = random_weights((num_output,), name=name+"_b_y")
        self.params = [self.W_yh]
	self.bias = [self.b_y]

    def output(self):
        return sigmoid(T.dot(self.X, self.W_yh) + self.b_y)

class DropoutLayer(NNLayer):

    def __init__(self, input_layer, is_train, name="",p=0.5):
        #is_train theano.iscalar symbolic variable indicating training/test
        self.X = input_layer.output()
        self.p = p
	self.name = name
        self.params = []
        self.is_train = is_train
	self.bias = []

    def output(self):
        return T.switch(T.neq(self.is_train, 0) , dropout(self.X,self.p), self.X)


class Squeeze(NNLayer):
	def __init__(self,input_layer,outdim=2,name=""):
		self.name = name
		self.params = []
		self.outdim= outdim
		self.X = input_layer.output()
		self.bias = []

	def output(self):
		return T.flatten(self.X, outdim=self.outdim)

class MergeLayer(NNLayer):
    def init(self, input_layers):
        return

    def output(self):
        return




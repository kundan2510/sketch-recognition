import numpy as np

import theano

import theano.tensor as T

class CostFunction:
	def __init__(self):
		pass


class CategoricalCrossEntropy(CostFunction):
	def __init__(self,output_layer,ground_truth,name=""):
		self.X = output_layer.output()
		self.Y = ground_truth
		self.name = name

	def output(self):
		return T.mean(T.nnet.categorical_crossentropy(self.X,self.Y))


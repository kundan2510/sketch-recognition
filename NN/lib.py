import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def random_weights(shape, name=None):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01), name=name, borrow=True)


def zeros(shape, name=""):
    return theano.shared(floatX(np.zeros(shape)), name=name, borrow=True)

def softmax(X, temperature=1.0):
    e_x = T.exp((X - X.max(axis=1).dimshuffle(0, 'x'))/temperature)
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def sigmoid(X):
    return T.nnet.sigmoid(X)

srng = RandomStreams(seed=123)

def dropout(X, p=0.):
	retain_prob = 1 - p
	X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
	X /= retain_prob
	return X

def rectify(X):
    return T.maximum(X, 0.)


def clip(X, epsilon):

    return T.maximum(T.minimum(X, epsilon), -1*epsilon)

def scale(X, max_norm):
    curr_norm = T.sum(T.abs_(X))
    return ifelse(T.lt(curr_norm, max_norm), X, max_norm * (X/curr_norm))

def SGD (cost, params, eta, lambda2 = 0.0):
    updates = []
    grads = T.grad(cost=cost, wrt=params)

    for p,g in zip(params, grads):
        updates.append([p, p - eta *( g + lambda2*p)])

    return updates

#rho*grad and not rho*g
def momentum(cost, params, biases,  caches_params, caches_biases, eta, rho=.1, clip_at=5.0, scale_norm=0.0, lambda2=0.001):
	updates = []
	grads_params = T.grad(cost=cost, wrt=params,disconnected_inputs='warn')
	grads_biases = T.grad(cost=cost, wrt=biases, disconnected_inputs='warn')
	for p, c, g in zip(params, caches_params, grads_params):
		if clip_at > 0.0:
			grad = clip(g, clip_at)
		else:
			grad = g
		if scale_norm > 0.0:
			grad = scale(g, scale_norm)

		delta = rho * g + (1-rho) * c
		updates.append([c, delta])
		updates.append([p, p - eta * ( delta + lambda2 * p)])

	for p, c, g in zip(biases, caches_biases, grads_biases):
		if clip_at > 0.0:
			grad = clip(g, clip_at)
		else:
			grad = g
		if scale_norm > 0.0:
			grad = scale(g, scale_norm)

		delta = rho * g + (1-rho) * c
		updates.append([c, delta])
		updates.append([p, p - eta * ( delta )])
		return updates


def get_params(layers):
    params = []
    for layer in layers:
        params += layer.get_params()
    return params

def get_biases(layers):
	biases = []
	for layer in layers:
		biases += layer.get_bias()

	return biases


def one_step_updates(layers):
    updates = []

    for layer in layers:
        updates += layer.updates()

    return updates	

def make_caches(params):
    caches = []
    for p in params:
        caches.append(theano.shared(floatX(np.zeros(p.get_value().shape))))

    return caches

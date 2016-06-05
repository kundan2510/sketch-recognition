from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, SGD
from keras.layers.recurrent import LSTM

def create_model():
	model = Sequential()

	model.add(Convolution2D(64, 15, 15, input_shape=(1,225,225)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

	model.add(Convolution2D(128, 5, 5))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))

	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))

	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

	model.add(Convolution2D(512, 4, 4))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Flatten())

	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(250))
	model.add(Activation('softmax'))

	#rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
	sgd = SGD(lr=0.0001, decay=1e-5, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd)

	return model

def create_CNN_LSTM():
	model = Sequential()

	model.add(Convolution2D(64, 15, 15, input_shape=(1,225,225), subsample=(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

	model.add(Convolution2D(128, 5, 5))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))

	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))

	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

	model.add(Convolution2D(512, 4, 4))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Flatten())

	model_lstm = Sequential()
	model_lstm.add(Merge([model,model,model,model,model], mode='concat'))
	model_lstm.add(Reshape((5,512)))
	model_lstm.add(LSTM(512,return_sequences=False))
	model_lstm.add(Dropout(0.5))
	model_lstm.add(Dense(250))
	model_lstm.add(Activation('softmax'))
	rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
	# sgd = SGD(lr=0.0001, decay=1e-5, momentum=0.9, nesterov=True)
	model_lstm.compile(loss='categorical_crossentropy', optimizer=rmsprop)
	return model_lstm

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, RemoteMonitor
from sys import argv
from data_loader_new import *

if __name__ == '__main__':
	assert(len(argv) >= 4)
	print "reading arguments"
	train_samples_file = argv[1]
	test_samples_file = argv[2]
	embedding_file = argv[3]
	print "Initializing data loading"
	#loader_train1 = data_loader(train_samples_file, from_chunk = False)
	#loader_train2 = data_loader(train_samples_file, from_chunk = False)
	loader_train = data_loader_new(train_samples_file, embedding_file=embedding_file)
	samples_train_count = loader_train.samples_count
	test_loader = data_loader_new(test_samples_file, embedding_file=embedding_file)
	samples_test_count = test_loader.samples_count
	print "Creating model...."
	model =  Sequential()
	model.add(GRU(250, input_shape=(5, 512), dropout_W=0.25, return_sequences=True))
	# model.add(Dropout(0.5))
	# model.add(LSTM(250, dropout_U=0.5, dropout_W=0.5))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	print "Loading data..."
	X_train,Y_train = loader_train.load_sequence_samples(num_elements = samples_train_count)
	X_test,Y_test = test_loader.load_sequence_samples(num_elements = samples_test_count)
 	model_check_pointing = ModelCheckpoint('../models/LSTM_weights.{epoch:03d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
	# rmm = RemoteMonitor(root='http://localhost:8080')
	model.fit(X_train,Y_train, batch_size=16, nb_epoch=2000, verbose=2, validation_data=(X_test,Y_test), callbacks=[model_check_pointing])
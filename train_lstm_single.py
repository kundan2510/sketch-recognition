from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, RemoteMonitor
from keras.regularizers import l2

from sys import argv
from data_loader_new import *

if __name__ == '__main__':
	assert(len(argv) >= 4)
	print "reading arguments"
	train_samples_file = argv[1]
	test_samples_file = argv[2]
	embedding_file = argv[3]
	num_per_seq = 3
	print "Initializing data loading"
	#loader_train1 = data_loader(train_samples_file, from_chunk = False)
	#loader_train2 = data_loader(train_samples_file, from_chunk = False)
	loader_train = data_loader_new(train_samples_file, embedding_file=embedding_file)
	samples_train_count = loader_train.samples_count
	test_loader = data_loader_new(test_samples_file, embedding_file=embedding_file)
	samples_test_count = test_loader.samples_count
	print "Creating model...."
	model =  Sequential()
	model.add(LSTM(256, input_shape=(num_per_seq, 512),dropout_U=0.50, dropout_W=0.50,return_sequences=True))
	model.add(Dropout(0.5))
	model.add(LSTM(128, dropout_U=0.50, dropout_W=0.50))
	model.add(Dropout(0.5))
	model.add(Dense(250))
	model.add(Activation('softmax'))
	
	json_string = model.to_json()
	open('model_squezzed_only_three_reg_256_128_ada.json', 'w').write(json_string)

	model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

	print "Loading data..."
	X_train,Y_train = loader_train.load_sequence_samples(num_elements = samples_train_count, num_per_seq=num_per_seq)
	X_test,Y_test = test_loader.load_sequence_samples(num_elements = samples_test_count, num_per_seq=num_per_seq)
 	model_check_pointing = ModelCheckpoint('../models/LSTM_weights_squeezed_only_three_reg_256_128_ada.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=False, mode='auto')
	# rmm = RemoteMonitor(root='http://localhost:8080')
	model.fit(X_train,Y_train, batch_size=200, nb_epoch=2000, verbose=1, validation_data=(X_test,Y_test), callbacks=[model_check_pointing])

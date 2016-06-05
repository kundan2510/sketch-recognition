from data_load_utils import *
from data_loader import *
from NN.CNN import *

loader = data_loader('train_samples.pkl',from_chunk=False)

X,Y = loader.load_samples(num_elements=8)

model = CNN(batch_size = 8)

for i in range(1000):
	loss, acc = model.train_on_batch(X,Y,0.01)
	if i+1 % 10: 
		print "epoch %d, accuracy %.4f, loss %.4f"%(i+1,acc,loss) 

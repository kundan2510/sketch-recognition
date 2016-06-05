import pickle
from random import shuffle
from data_load_utils import *

class data_loader:

	def __init__(self, samples_file, from_chunk=True,transform=False):
		self.transform = transform
		if not from_chunk:
			self.from_chunk = False

			self.samples = pickle.load(open(samples_file,'rb'))

			self.next_sample_from_class = {}

			self.total_sample_count_classes = {}

			self.samples_count = 0

			self.classes = self.samples.keys()

			self.next_class = 0
			
			self.done_so_far = 0			

			self.class_count = len(self.classes)

			for key in self.classes:
				shuffle(self.samples[key])
				self.next_sample_from_class[key] = 0
				self.total_sample_count_classes[key] = len(self.samples[key])
				self.samples_count += len(self.samples[key])
		else:
			self.from_chunk = True
			self.chunk_files = pickle.load(open(samples_file,'rb'))
			self.num_chunks = len(self.chunk_files['X'])
			self.next_chunk_num = 0


	def next_sample(self):
		assert(not self.from_chunk)
		sample = self.samples[ self.classes[self.next_class] ][ self.next_sample_from_class[self.next_class] ]
		self.next_sample_from_class[self.next_class] = (self.next_sample_from_class[self.next_class] + 1)%self.total_sample_count_classes[self.next_class]
		self.next_class = (self.next_class+1)%self.class_count
		self.done_so_far = (self.done_so_far + 1)%self.samples_count
		if self.done_so_far == 0:
			for key in self.classes:
                                shuffle(self.samples[key])
		return sample

	def load_batch_file_list(self,num_elements):
		assert(not self.from_chunk)

		batch_files_list = []

		for i in range(num_elements):
			batch_files_list.append(self.next_sample())

		return batch_files_list

	def file_list_to_XY(self,batch_files_list):
		X = []
		Y = []
		for i,f in enumerate(batch_files_list):
			f_name = f.split('#')[0]
			cl = int(f.split('#')[2])
			x,y = image_to_sample(f_name,cl,transform=self.transform)
			X.append(x)
			Y.append(y)
			# print "Done %d"%i

		X = np.asarray(X)
		Y = np.asarray(Y)

		return X,Y

	def load_samples(self,num_elements=250):
		batch_files_list = self.load_batch_file_list(num_elements)
		return self.file_list_to_XY(batch_files_list)

	def make_chunks(self,num_chunks, base_file_name='train_'):
		elements_per_chunk = self.samples_count/num_chunks

		chunk_names_dict = {}
		chunk_names_dict['X'] = []
		chunk_names_dict['Y'] = []

		if self.samples_count % num_chunks == 0:
			chunks_sizes = [elements_per_chunk]*num_chunks
		else:
			chunks_sizes = [elements_per_chunk]*(num_chunks-1) + [elements_per_chunk + self.samples_count%num_chunks]

		for i,chunk_size in enumerate(chunks_sizes):
			X,Y = self.load_samples(num_elements=chunk_size)

			fX_name = base_file_name+'_X_'+str(i)+'_'+str(chunk_size)+'.npy'
			fX = open(fX_name,'wb')
			chunk_names_dict['X'].append(fX_name)
			np.save(fX,X)

			fY_name = base_file_name+'_Y_'+str(i)+'_'+str(chunk_size)+'.npy'
			fY = open(fY_name,'wb')
			chunk_names_dict['Y'].append(fY_name)

			np.save(fY,Y)
			print "Done %s"%i

		pickle.dump(chunk_names_dict,open(base_file_name+'_chunk_names_dict.pkl','wb'))

	def next_chunk(self):
		assert(self.from_chunk)
		X = np.load(self.chunk_files['X'][self.next_chunk_num])
		Y = np.load(self.chunk_files['Y'][self.next_chunk_num])
		self.next_chunk_num = (self.next_chunk_num + 1)%self.num_chunks
		return X,Y



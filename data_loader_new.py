from random import shuffle
from data_load_utils import *
import numpy as np
from pre_trained_embedd import *
from image_transform import *

class data_loader_new:
	def __init__(self, file_name, shfl=True, embedding_file= None, transform_prob=0.99):
		self.image_list = []
		with open(file_name,'rb') as f:
			for row in f:
				self.image_list.append(row)
		if shfl:
			shuffle(self.image_list)
		self.shfl = shfl
		self.samples_count = len(self.image_list)

		self.done_so_far = 0
		self.img_transformer = ImageTransform(rotRange=20,hshift=0.15,vshift=0.15,hflip=True,vflip=False,shearAngle=3,fill_mode="nearest",cval=0.0, p_not_transform=(1- transform_prob ))

		if embedding_file is not None:
			self.embedding = True
			self.embedder = image_embedder(embedding_file=embedding_file)
		else:
			self.embedding = False


	def next_sample_name_and_label(self):
		self.done_so_far += 1
		if self.done_so_far == self.samples_count:
			#print "I am here with don %d and samples count"%(self.done_so_far, self.samples_count)
			self.done_so_far = 0
			if self.shfl:
				shuffle(self.image_list)
		return self.image_list[self.done_so_far]

	def load_samples(self,num_elements=250,transform=False):
		X = np.zeros((num_elements,1,225,225),dtype=np.float32)
		Y = np.zeros((num_elements,250),dtype=np.float32)
		for i in range(num_elements):
                	example_name_label = self.next_sample_name_and_label().strip('\n')
			split_name = example_name_label.split('#')
			img_file_name = split_name[0]
			label = int(split_name[2])
			x, y = image_to_sample(img_file_name,label,transform=transform)

			if transform:
				x = self.img_transformer.random_transform(x)
			#print self.samples_count
			# print self.done_so_far
			#print label
			#print np.argmax(y)
			X[i] = x
			Y[i] = y
		#print(Y[0:1])	
		return X, Y

	def load_sequence_samples(self,num_elements=50,num_per_seq = 5, transform=False):
		if self.embedding == False:
			X = np.zeros((num_elements,num_per_seq,1,225,225),dtype=np.float32)
		else:
			global image_to_sample
			image_to_sample = self.embedder.load_image
			X = np.zeros((num_elements,num_per_seq,512),dtype=np.float32)
		Y = np.zeros((num_elements,250),dtype=np.float32)
		for i in range(num_elements):
			example_name_label = self.next_sample_name_and_label().strip('\n')
			split_name = example_name_label.split('#')
			# img_file_name1 = split_name[0]; img_file_name2 = split_name[1];
			# img_file_name3 = split_name[2];img_file_name4 = split_name[3];
			# img_file_name5 = split_name[4];
			# label = int(split_name[5])
			# x1, y = image_to_sample(img_file_name1,label,transform=transform)
			# x2, y = image_to_sample(img_file_name2,label,transform=transform)
			# x3, y = image_to_sample(img_file_name3,label,transform=transform)
			# x4, y = image_to_sample(img_file_name4,label,transform=transform)
			# x5, y = image_to_sample(img_file_name5,label,transform=transform)
			# #print self.samples_count
			# #print self.done_so_far
			# #print label
			# #print np.argmax(y)
			# X[i,0] = x1; X[i,1] = x2; X[i,2] = x3; X[i,3] = x4; X[i,4] = x5;
			# Y[i] = y
			label = int(split_name[5])
			for j in range(num_per_seq):
				x,y = image_to_sample(split_name[5- num_per_seq + j],label,transform=transform)
				X[i,j] = x
				Y[i] = y
		#print(Y[0:1])	
		return X, Y

from skimage import io
from skimage.transform import resize

from PIL import Image

#import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from os.path import isfile, join, isdir

import cPickle as pickle


from image_transform import *

uid = 0

image_data_set = {}

def read_image(fname):
	return Image.open(fname)

def resize_image(img, output_shape=(225,225)):
	return img.resize(output_shape, Image.ANTIALIAS)

def image_to_sample(f_name,index,transform=False,save=False):
	y = np.zeros(250,dtype=np.float32)
	y[index] = 1.0
	global image_data_set
	if f_name in image_data_set:
		x_new = image_data_set[f_name]
	else:
		img = read_image(f_name)
		img = resize_image(img)
		bg = Image.new("RGB", img.size, (255,255,255))
		bg.paste(img,img)
		bg = bg.convert('L')
		x = ((np.asarray(bg)).astype(np.float32))/255.0
		#r = x[:,:,0]
		#g = x[:,:,1]
		#b = x[:,:,2]
		x_new = np.zeros((1,np.shape(x)[0],np.shape(x)[1]),dtype=np.float32)
		x_new[0] = x
		image_data_set[f_name] = x_new
	# global uid
	# # if transform:
	# # 	x_new = img_transformer.random_transform(x_new)
	# # if save:
	# # 	img_transformer.save_image(x_new,"../data/augmented/"+f_name.split("/")[-2]+"_"+f_name.split("/")[-1].split('.')[0]+str(uid)+".jpg")
	# uid += 1
	#x_new[0] = r
	#x_new[1] = g
	#x_new[2] = b
	return x_new,y


def image_to_sample_class_name(f_name,class2ind,d):
	index = y[class2ind[d]]
	return image_to_sample_class_index(f_name,index)

def create_image_list(image_folder_path = '/home/kundan/data/Secondtry', index_in_sequence = 4, output_folder='/home/kundan/code/description_files/'):
	class2ind = {}
	ind2class = {}
	#list of all directories in folder path
	dirs = [o for o in listdir(image_folder_path)]

	assert(len(dirs) == 250)

	output_train_file = output_folder + 'img_train_file_second_'+str(index_in_sequence)+'.txt'
	output_test_file = output_folder + 'img_test_file_second_'+str(index_in_sequence)+'.txt'
	out_train_f = open(output_train_file,'wb')
	out_test_f = open(output_test_file, 'wb')

	X = []
	Y = []

	class_files_dict = {}
	index_files_dict = {}

	for i,d in enumerate(dirs):
		class2ind[d] = i
		ind2class[i] = d
		class_files_dict[d] = []
		index_files_dict[i] = []

	for d in dirs:
		curr_dir = join(image_folder_path,d)
		j = 0
		for f in listdir(curr_dir):
			f_name = join(curr_dir,f)
			if isfile(f_name) and (('svg_'+str(index_in_sequence)) in f_name):
				#print "Loading %s from %s"%(f_name,d)
				singe_sample = f_name + '#'+d + '#'+str(class2ind[d])
				class_files_dict[d].append(singe_sample)
				index_files_dict[class2ind[d]].append(singe_sample)
				if j < 55:
					out_train_f.write(singe_sample+'\n')
				else:
					out_test_f.write(singe_sample+'\n')
				j += 1
		print "Done for directory %s"%d
	out_train_f.close()
	out_test_f.close()
	# pickle.dump(class2ind,open(class2ind_file,'wb'))
	# pickle.dump(class_files_dict,open(class_dict,'wb'))
	# pickle.dump(index_files_dict,open(index_sample_list,'wb'))

def create_image_list_custom(image_folder_path = '/home/kundan/data/Secondtry', indices_in_sequence = [4], output_folder='/home/kundan/code/description_files/'):
	class2ind = {}
	ind2class = {}
	#list of all directories in folder path
	dirs = [o for o in listdir(image_folder_path)]

	assert(len(dirs) == 250)

	indices_name = ""
	indices_in_sequence = np.asarray(indices_in_sequence)

	for ind in indices_in_sequence:
		indices_name += str(ind)+"_"

	output_train_file = output_folder + 'img_train_file_second_'+indices_name+'.txt'
	output_test_file = output_folder + 'img_test_file_second_'+indices_name+'.txt'
	out_train_f = open(output_train_file,'wb')
	out_test_f = open(output_test_file, 'wb')

	X = []
	Y = []

	class_files_dict = {}
	index_files_dict = {}

	for i,d in enumerate(dirs):
		class2ind[d] = i
		ind2class[i] = d
		class_files_dict[d] = []
		index_files_dict[i] = []

	for d in dirs:
		curr_dir = join(image_folder_path,d)
		count = np.zeros(len(indices_in_sequence))
		for f in listdir(curr_dir):
			f_name = join(curr_dir,f)
			if isfile(f_name):
				curr_ind = -1
				for ind in indices_in_sequence:
					if (('svg_'+str(ind)) in f_name):
						curr_ind = ind
						break

				if curr_ind < 0:
					continue
				#print "Loading %s from %s"%(f_name,d)
				singe_sample = f_name + '#'+d + '#'+str(class2ind[d])
				class_files_dict[d].append(singe_sample)
				index_files_dict[class2ind[d]].append(singe_sample)
				count_curr_ind = count[np.where(indices_in_sequence==curr_ind)[0][0]]
				#if count_curr_ind >= 27:
					#out_train_f.write(singe_sample+'\n')
				#else:
				#	if curr_ind == 4:
				#		out_test_f.write(singe_sample+'\n')
				#count[np.where(indices_in_sequence==curr_ind)[0][0]] += 1
		print "Done for directory %s"%d

	for cl in index_files_dict:
		class_examples = sorted(index_files_dict[cl])
		last_example = {}
		for f in class_examples:
			if 'svg_4' in f:
				last_example[int(f.split('.')[0].split('/')[-1])] = 1
		test_example = {}
		for i,j in enumerate(last_example.keys()):
			if i >= 54:
				test_example[j] = 1

		for f in class_examples:
                        if 'svg_4' in f and int(f.split('.')[0].split('/')[-1]) in test_example:
				out_test_f.write(f+'\n')
			elif int(f.split('.')[0].split('/')[-1]) not in test_example:
				out_train_f.write(f+'\n')
		
	out_train_f.close()
	out_test_f.close()
	# pickle.dump(class2ind,open(class2ind_file,'wb'))
	# pickle.dump(class_files_dict,open(class_dict,'wb'))
	# pickle.dump(index_files_dict,open(index_sample_list,'wb'))

def create_image_sequence_list(image_folder_path = '/home/kundan/data/Secondtry',indices_in_sequence = [0,1,2,3,4], output_folder='/home/kundan/code/description_files/'):
	class2ind = {}
	ind2class = {}
	#list of all directories in folder path
	dirs = [o for o in listdir(image_folder_path)]

	assert(len(dirs) == 250)

	indices_name = ""
	indices_in_sequence = np.asarray(indices_in_sequence)

	for ind in indices_in_sequence:
		indices_name += str(ind)+"_"

	output_train_file = output_folder + 'img_train_seq_file_second_'+indices_name+'.txt'
	output_test_file = output_folder + 'img_test_seq_file_second_'+indices_name+'.txt'
	out_train_f = open(output_train_file,'wb')
	out_test_f = open(output_test_file, 'wb')

	X = []
	Y = []

	class_files_dict = {}
	index_files_dict = {}

	for i,d in enumerate(dirs):
		class2ind[d] = i
		ind2class[i] = d
		class_files_dict[d] = []
		index_files_dict[i] = []

	for d in dirs:
		curr_dir = join(image_folder_path,d)
		count = np.zeros(len(indices_in_sequence))
		for f in listdir(curr_dir):
			f_name = join(curr_dir,f)
			if isfile(f_name):
				curr_ind = -1
				for ind in indices_in_sequence:
					if (('svg_'+str(ind)) in f_name):
						curr_ind = ind
						break

				if curr_ind < 0:
					continue
				#print "Loading %s from %s"%(f_name,d)
				singe_sample = f_name #+ '#'+d + '#'+str(class2ind[d])
				class_files_dict[d].append(singe_sample)
				index_files_dict[class2ind[d]].append(singe_sample)
				# count_curr_ind = count[np.where(indices_in_sequence==curr_ind)[0][0]]
				#if count_curr_ind >= 27:
					#out_train_f.write(singe_sample+'\n')
				#else:
				#	if curr_ind == 4:
				#		out_test_f.write(singe_sample+'\n')
				#count[np.where(indices_in_sequence==curr_ind)[0][0]] += 1
		print "Done for directory %s"%d

	for cl in index_files_dict:
		class_examples = sorted(index_files_dict[cl])
		id_wise_examples = {}
		for f in class_examples:
			if 'svg' in f:
				if int(f.split('.')[0].split('/')[-1]) not in id_wise_examples:
					id_wise_examples[int(f.split('.')[0].split('/')[-1])] = [f]
				else:
					id_wise_examples[int(f.split('.')[0].split('/')[-1])] += [f]

		test_example = {}
		for i,j in enumerate(id_wise_examples.keys()):
			examples = sorted(id_wise_examples[j])
			if i >= 54:
				for ex in examples:
					out_test_f.write(ex+"#")
				out_test_f.write(str(cl)+'\n')
			else:
				for ex in examples:
					out_train_f.write(ex+"#")
				out_train_f.write(str(cl)+'\n')
	
	out_train_f.close()
	out_test_f.close()


def make_train_test(ind_sample,train_ratio = 0.67):
	train_samples = {}
	test_samples = {}
	for index in ind_sample:
		num2take = int(len(ind_sample[index])*0.67)
		train_samples[index] = ind_sample[index][:num2take]
		test_samples[index] = ind_sample[index][num2take:]
	pickle.dump(train_samples,open('train_samples.pkl','wb'))
	pickle.dump(test_samples,open('test_samples.pkl','wb'))

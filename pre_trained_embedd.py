import cPickle as pickle
from data_load_utils import *
import numpy as np

class image_embedder:
	def __init__(self, img_files_list = None, embedding_file=None,network=None):

		if embedding_file is None:
			assert network is not None
			self.CNN = network
			assert img_files_list is not None
			''' img_files_list will be a tuple of train and test image file list'''
			train_list_file = img_files_list[0]
			test_list_file = img_files_list[1]
			img_list = []
			with open(train_list_file,'rb') as tf:
				for row in tf:
					img_list.append(row)
			
			with open(test_list_file,'rb') as tf:
                                for row in tf:
                                        img_list.append(row)

			embedding_file = self.img_list_to_embeddings(img_list)

		self.embeddings = pickle.load(open(embedding_file,'rb'))

	def img_list_to_embeddings(self, img_list):
		i = 0
		to_save = {}
		output_pkl = 'description_files/image_embeddings_final_second_2_3_4.pkl'
		for sample in img_list:
			split_name = sample.strip('\n').split('#')
                        img_file_name1 = split_name[0]; img_file_name2 = split_name[1];
                        img_file_name3 = split_name[2];img_file_name4 = split_name[3];
                        img_file_name5 = split_name[4];
                        label = int(split_name[5])
                        x1, y = image_to_sample(img_file_name1,label,transform=False)
                        x2, y = image_to_sample(img_file_name2,label,transform=False)
                        x3, y = image_to_sample(img_file_name3,label,transform=False)
                        x4, y = image_to_sample(img_file_name4,label,transform=False)
                        x5, y = image_to_sample(img_file_name5,label,transform=False)
			
			data = np.asarray([x1,x2,x3,x4,x5])

			embeddings = self.CNN.get_embeddings(data,0)

			to_save[img_file_name1] = {'embed': embeddings[0],'y':y}

			to_save[img_file_name2] = {'embed': embeddings[1],'y':y}

			to_save[img_file_name3] = {'embed': embeddings[2],'y':y}

			to_save[img_file_name4] = {'embed': embeddings[3],'y':y}

			to_save[img_file_name5] = {'embed': embeddings[4],'y':y}
			i += 1
			print "Done %d"%(i)
			
		pickle.dump(to_save, open(output_pkl,'wb'))

					
		return output_pkl
				

	def load_image(self, image_name,label,transform=False):
		return self.embeddings[image_name]['embed'], self.embeddings[image_name]['y'] 



import numpy as np

class running_average:
	def __init__(self,size):
		self.size = size
		self.next_pos = 0
		self.num_elements = 0
		self.arr = np.zeros(size)

	def insert(self,element):
		self.arr[self.next_pos] = element
		self.next_pos = (self.next_pos + 1)%self.size
		if self.num_elements < self.size:
			self.num_elements += 1

	def average(self):
		assert(self.num_elements > 0)
		return np.sum(self.arr)/self.num_elements

	def upsert(self,element):
		self.insert(element)
		return self.average()
		
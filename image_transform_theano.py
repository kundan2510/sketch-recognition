from scipy import ndimage
import numpy as np
import random
import math
from PIL import Image


#TODO Make it run fast

class ImageTransform:
	def __init__(self,rotRange=0,hshift=0,vshift=0,hflip=False,vflip=False,shearAngle=0,fill_mode="nearest",cval=0.0,p_not_transform=0.01):
		self.__dict__.update(locals())
		self.not_transform_prob = np.ones(6)*p_not_transform
		self.not_transform_prob[4] = 1.0
		self.not_transform_prob[5::] = 1.0

	def random_rotation(self,x,rg,fill_mode="nearest",cval=0.0):
		angle = random.uniform(-rg,rg)
		x = ndimage.interpolation.rotate(x,angle,axes=(1,2),reshape=False,mode=fill_mode,cval=cval)
		return x

	def random_hshift(self,x,hrg,fill_mode="nearest",cval=0.0):
		crop = random.uniform(0,hrg)
		split=random.uniform(0,1)
		crop_pixels = int(split*crop*x.shape[2])
		x=ndimage.interpolation.shift(x,(0,0,crop_pixels),order=0,mode=fill_mode,cval=cval)
		return x

	def random_vshift(self,x,wrg,fill_mode="nearest",cval=0.0):	
		crop = random.uniform(0,wrg)
		split=random.uniform(0,1)
		crop_pixels = int(split*crop*x.shape[1])
		x=ndimage.interpolation.shift(x,(0,crop_pixels,0),order=0,mode=fill_mode,cval=cval)
		return x

	def horizontal_flip(self,x):
		for i in range(x.shape[0]):
			x[i] = np.fliplr(x[i])
		return x


	def vertical_flip(self,x):
		for i in range(x.shape[0]):
			x[i] = np.flipud(x[i])
		return x

	def random_shear(self,x,srg,fill_mode="nearest",cval=0.0):
		shear = random.uniform(-srg,srg)
		shear_matrix = np.array([[1.0,-math.sin(shear),0.0],[0.0,math.cos(shear),0.0],[0.0,0.0,1.0]])
		x = ndimage.interpolation.affine_transform(x,shear_matrix,mode=fill_mode,order=3,cval=cval)
		return x

	def random_zoom(self,x,zrg,fill_mode="nearest",cval=0.0):
		zoom_w=random.uniform(1.0-zrg,1.0)
		zoom_h=random.uniform(1.0-zrg,1.0)
		x=ndimage.interpolation.zoom(x,zoom=(1.0,zoom_w,zoom_h),mode=fill_mode,cval=cval)
		return x

	def random_transform(self,x):
		on = np.random.uniform(0,1.0,6) > self.not_transform_prob
		if self.rotRange and on[0]:
			x=self.random_rotation(x,self.rotRange)
		if self.hshift and on[1]:
			x=self.random_hshift(x,self.hshift)
		if self.vshift and on[2]:
			x=self.random_vshift(x,self.vshift)
		if self.hflip and on[3]:
			x=self.horizontal_flip(x)
		if self.vflip and on[4]:
			x=self.vertical_flip(x)
		if self.shearAngle and on[5]:
			x=self.random_shear(x,self.shearAngle)
		return x

	def arr_to_image(self,x,scale=True):
		x = x.transpose(1,2,0)
		if scale:
			x*=255.0
		if x.shape[2]==3:#RGB
			return Image.fromarray(x.astype("uint8"),"RGB")
		else:
			return Image.fromarray(x[:,:,0].astype("uint8"),"L")

	def save_image(self,x,path):
		im=self.arr_to_image(x)
		im.save(path)
		

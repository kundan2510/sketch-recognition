from image_transform import *
import sys
from PIL import Image
import numpy as np

def load_image_as_array(path,gray=True):
	img = Image.open(path)
	if gray:
		img=img.convert("L")
	
	return image2array(img)

def image2array(img):
	x=np.asarray(img,dtype='float32')
	if len(x.shape)==3:
		x=x.transpose(2,0,1)
	else:
		x=x.reshape((1,x.shape[0],x.shape[1]))
	return x

transformer = ImageTransform()
filePath = sys.argv[1]
x = load_image_as_array(filePath)
transformer.save_image(transformer.random_rotation(x,10),"temp/imgRot.jpg")
x = load_image_as_array(filePath)
transformer.save_image(transformer.random_hshift(x,0.3),"temp/imgHshift.jpg")
x = load_image_as_array(filePath)
transformer.save_image(transformer.random_vshift(x,0.3),"temp/imgVshift.jpg")
x = load_image_as_array(filePath)
transformer.save_image(transformer.horizontal_flip(x),"temp/imghflip.jpg")
x = load_image_as_array(filePath)
transformer.save_image(transformer.vertical_flip(x),"temp/imgvflip.jpg")
x = load_image_as_array(filePath)
transformer.save_image(transformer.random_shear(x,10),"temp/imgShear.jpg")





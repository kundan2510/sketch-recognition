from utils import *
import sys
import argparse
import os
import cairosvg

dataPath = "/home/sanilj/Downloads/SketchANet/data"
parser = argparse.ArgumentParser()
parser.add_argument("-n","--nSteps",help="Number of steps in which to break the image",type=int, default=5)
parser.add_argument("-t","--target",help="target Directory Name | it will be created in data folder")
args = parser.parse_args()

if not args.target:
		raise Exception("Target Directory Must be Specified")


def getImageSvg(strokeLines,endStep):
	newSvg = "".join(strokeLines[:5+endStep+1]) #Starting code+path lines
	newSvg += "".join(strokeLines[-3:])
	return newSvg

def createPngSplit(image,categDir,origDir,nsteps):
	strokeLines = getStrokeLines(os.path.join(origDir,image))
	strokeCount = getStrokeCount(strokeLines)
	steps=[]
	if strokeCount<=nsteps:
		steps=range(1,strokeCount)+[strokeCount-1]*(nsteps-strokeCount+1)
	else:
		steps = range(0,strokeCount,strokeCount/nsteps)[1:nsteps+1] #TO DO, Can change algo?
		if len(steps)<nsteps:
			steps.append(strokeCount-1)
		steps[-1]=max(steps[-1],strokeCount-1) #-1 since indexing from 0
	for i,step in enumerate(steps):
		imageSvg = getImageSvg(strokeLines,step)
		imageFile = open(os.path.join(categDir,image+"_"+str(i)+".png"),"w")
		cairosvg.svg2png(bytestring=imageSvg,write_to=imageFile)
		imageFile.close()
	
def svgToPng():
	imageDic = createImageDict(imageList)
	targetDir = args.target
	os.mkdir(os.path.join(dataPath,targetDir))
	for category in imageDic.keys():
		categDir = os.path.join(dataPath,targetDir,category)
		origDir = os.path.join(dataPath,"svg",category)
		os.mkdir(categDir)
		for image in imageDic[category]:
			createPngSplit(image,categDir,origDir,args.nSteps)



svgToPng()
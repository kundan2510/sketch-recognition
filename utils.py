import os

def getStrokeLines(fileName):
	print "fileName:"+fileName
	lines = [line.strip() for line in open(fileName,'rb').readlines()]
	return lines

def getStrokeCount(strokeLines):
	return len(strokeLines)-5-3#start and ending lines

#imageList = [line.strip() for line in open("/home/sanilj/Downloads/SketchANet/data/svg/filelist.txt").readlines()]

imageList = [line.strip() for line in open("/home/sanilj/Downloads/SketchANet/data/svg/filelist.txt").readlines()]

def createImageDict(imageList):
	imageDic = {}
	for image in imageList:
		category,imageName = image.split('/')
		if not category in imageDic:
			imageDic[category]=[]
		else:
			imageDic[category].append(imageName)
	return imageDic

def getMinStrokeCount():
	imageDic = createImageDict(imageList)
	minStroke = 1e10
	minStrokeFile = ""
	rootPath = "/home/sanilj/Downloads/SketchANet/data/svg"
	for category in imageDic.keys():
		for imageNum in imageDic[category]:
			filePath = os.path.join(rootPath,category,imageNum)
			strokeCount = getStrokeCount(filePath)
			if minStroke > strokeCount:
				minStroke=strokeCount
				minStrokeFile = filePath
	return minStroke,minStrokeFile

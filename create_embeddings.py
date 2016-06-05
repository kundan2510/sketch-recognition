from NN.CNN import *

from sys import argv

network = CNN(batch_size=5)

network.load_model_params_dumb(argv[1])

from pre_trained_embedd import *

embedder = image_embedder(img_files_list=argv[2::], network=network)

#hello world

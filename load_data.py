import numpy as np 
import pickle
import os

from utils import loadimgs


train_folder = "./data/images_background/"
val_folder = './data/images_evaluation/'
save_path = './data/'

X,y,c=loadimgs(train_folder)

with open(os.path.join(save_path,"train.pickle"), "wb") as f:
    pickle.dump((X,c),f)


Xval,yval,cval=loadimgs(val_folder)

with open(os.path.join(save_path,"val.pickle"), "wb") as f:
    pickle.dump((Xval,cval),f)
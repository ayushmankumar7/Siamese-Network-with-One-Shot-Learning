import numpy as np 
import tensorflow as tf 


def initialize_weights(shape, dtype = "", name = ""):
     
     return np.random.normal(loc= 0.0, scale= 0.01, size = shape)


def initialize_bias(shape,dtype = "", name = ""):
    
    return np.random.normal(loc= 0.0, scale= 0.01, size=shape)


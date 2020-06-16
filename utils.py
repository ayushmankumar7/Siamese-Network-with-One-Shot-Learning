import numpy as np 
import tensorflow as tf 


def initialize_weights(shape):
     
     return np.random.normal(loc= 0.0, scale= 0.01, size = shape)


def initialize_bias(shape):
    
    return np.random.normal(loc= 0.0, scale= 0.01, size=shape)


import tensorflow as tf 
import numpy as np 


from model import siamese_network


model = siamese_network((105,105, 1))

print(model.summary())


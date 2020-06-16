import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Lambda
from tensorflow.keras.models import Model, Sequential

# tf.keras.initializers.lecun_normal()

def siamese_network(input_shape):

    input1 = Input(input_shape)
    input2 = Input(input_shape)


    model = Sequential([

        Conv2D(64, (10,10), activation = 'relu', input_shape = input_shape , kernel_initializer = initialize_weights, kernel_regulatizer = tf.keras.regularizers.l2(2e-4)),
        MaxPooling2D(),
        Conv2D(128, (7,7), activation = 'relu', kernel_initializer = initialize_weights, bias_initializer = initialize_bias, kernel_regulatizer = tf.keras.regularizers.l2(2e-4)),
        MaxPooling2D(),
        Conv2D(128, (4,4), activation = 'relu', kernel_initializer = initialize_weights, bias_initializer = initialize_bias, kernel_regulatizer = tf.keras.regularizers.l2(2e-4)),
        MaxPooling2D(),
        Conv2D(256, (4,4), activation = 'relu', kernel_initializer = initialize_weights, bias_initializer = initialize_bias, kernel_regulatizer = tf.keras.regularizers.l2(2e-4)),
        Flatten(),
        Dense(4096, activation = 'sigmoid', kernel_initializer = initialize_weights, bias_initializer = initialize_bias, kernel_regulatizer = tf.keras.regularizers.l2(1e-3))
    ])

    encoded_1 = model(input1)
    encoded_2 = model(input2)

    L1_layer = Lambda(lambda x: abs(x[0] - x[1]))
    L1_distance = L1_layer([encoded_1, encoded_2])

    prediction = Dense(1, activation = 'sigmoid')(L1_distance)

    final_model = Model(inputs = [input1, input2], outputs = prediction)


    return final_model
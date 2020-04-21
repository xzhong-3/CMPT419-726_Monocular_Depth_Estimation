# Using keras to implement Alexnet
import numpy as np 
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
import keras.backend as K
import cv2


def L7(prev_disparity_tens,h_grad_tens,rescaled_left_image_tens,prev_warped_image):

    def loss_fn(conv7,prev_warped_image):

        loss_value = K.square((((conv7-prev_disparity_tens)*h_grad_tens)+prev_warped_image)-rescaled_left_image_tens)

        return loss_value


    # Defining the optimizer
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    # Define the input
    inputs = Input(shape=(188,620,3))

    # Output from first convolutional layer
    conv1 = Conv2D(filters=96, kernel_size=(11,11), activation='relu', strides=(4,4), padding='same',kernel_initializer = 'he_normal')(inputs)
    conv1_mp = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv1)

    # Output from second convolutional layer
    conv2 = Conv2D(filters=256, kernel_size=(11,11), activation='relu', strides=(1,1), padding='same',kernel_initializer = 'he_normal')(conv1_mp)
    conv2_mp = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv2)

    # Output from third convolutional layer
    conv3 = Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='same',kernel_initializer = 'he_normal')(conv2_mp)
    
    # Output from fourth convolutional layer
    conv4 = Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='same',kernel_initializer = 'he_normal')(conv3)

    # Output from fifth convolutional layer
    conv5 = Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), padding='same',kernel_initializer = 'he_normal')(conv4)
    conv5_mp = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv5)

    # Output from the middle layer
    conv6 = Conv2D(filters=2048, kernel_size=(5,5), activation='relu', strides=(1,1), padding='same',kernel_initializer = 'zeros')(conv5_mp)

    # Output from layer 7 (should only be one image)
    conv7 = Conv2DTranspose(filters=3, kernel_size=(1,1), strides=(1, 1), padding='valid',  activation='relu', use_bias=True, kernel_initializer='zeros')(conv6)

    

    # Define the inputs and outputs of the model

    model = Model(inputs = inputs, outputs = conv7)
    model.compile(optimizer = sgd, loss = loss_fn)

    #model.summary()

    return model
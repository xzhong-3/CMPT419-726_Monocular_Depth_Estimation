import numpy as np 
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import initializers
import tensorflow as tf
import keras.backend as K
import cv2
import gc
import math

def L9_validation(model_weights, left_valid_lst, right_valid_lst):

    # Define model 9
    inputs = Input(shape=(188,620,3))
    # Output from first convolutional layer
    conv1 = Conv2D(filters=96, kernel_size=(11,11), activation='relu', strides=(4,4), padding='same',kernel_initializer = 'he_normal',kernel_regularizer=l2(0.0005))(inputs)
    conv1_mp = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv1)
    # Output from second convolutional layer
    conv2 = Conv2D(filters=256, kernel_size=(11,11), activation='relu', strides=(1,1), padding='same',kernel_initializer = 'he_normal',kernel_regularizer=l2(0.0005))(conv1_mp)
    conv2_mp = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv2)
    # Output from third convolutional layer
    conv3 = Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='same',kernel_initializer = 'he_normal',kernel_regularizer=l2(0.0005))(conv2_mp)
    # Output from fourth convolutional layer
    conv4 = Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='same',kernel_initializer = 'he_normal',kernel_regularizer=l2(0.0005))(conv3)
    # Output from fifth convolutional layer
    conv5 = Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), padding='same',kernel_initializer = 'he_normal',kernel_regularizer=l2(0.0005))(conv4)
    conv5_mp = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv5)
    # Output from the middle layer
    conv6 = Conv2D(filters=2048, kernel_size=(5,5), activation='relu', strides=(1,1), padding='same',kernel_initializer = initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None),kernel_regularizer=l2(0.0005))(conv5_mp)
    # Output from layer 7 (should only be one image)
    conv7 = Conv2DTranspose(filters=1, kernel_size=(1,1), strides=(1, 1), padding='valid',  activation='relu', kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None),kernel_regularizer=l2(0.0005))(conv6)
    # Bilinear upsampling from layer 7 to 8 and cropping the upsampled tensor to reduce by one column
    c7_up = Cropping2D(cropping=((0,0),(0,1)))(UpSampling2D(size=(2,2))(conv7))
    # Convolve the upsampled results
    c7_up_conv = Conv2D(filters=1, kernel_size=(4,4), activation='relu', strides=(1,1), padding='same',kernel_initializer = initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None),kernel_regularizer=l2(0.0005))(c7_up)
    # Defining a tranpose convolution for layer 5 to produce one output tensor to be merged with upsampled and corpped layer 8
    T_conv5 = Conv2DTranspose(filters=1, kernel_size=(1,1), strides=(1, 1), padding='valid',  activation='relu', kernel_initializer=initializers.RandomUniform(minval=-0.002, maxval=0.002, seed=None),kernel_regularizer=l2(0.0005))(conv5)
    # Merging the layers
    merged_layer8 = Add()([T_conv5,c7_up_conv])
    # Bilinear upsampling from Layer8 to 9
    conv9 = UpSampling2D(size=(2,2))(merged_layer8)
    # Convolve the output of layer 9
    c9_conv = Conv2D(filters=1, kernel_size=(4,4), activation='relu', strides=(1,1), padding='same',kernel_initializer = initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None),kernel_regularizer=l2(0.0005))(conv9)
    # Defining a tranpose convolution that squashes the 3rd dimension of the ouput of layer 3
    T_conv2 = Conv2DTranspose(filters=1, kernel_size=(1,1), strides=(1, 1), padding='valid',  activation='relu', kernel_initializer=initializers.RandomUniform(minval=-0.002, maxval=0.002, seed=None),kernel_regularizer=l2(0.0005))(conv2)
    # Merging the layers
    merge_layer_9 = Add()([T_conv2,c9_conv])
    # Defining the new output of the model
    model = Model(inputs = inputs, outputs = merge_layer_9)

    # Load the model weights
    model.set_weights(model_weights)

    # Given all of the left images in the provided validation list, predict the disparity map given the left image.
    left_validation_lst = left_valid_lst
    right_validation_lst = right_valid_lst

    # Variable to store total validation loss
    validation_loss = 0

    # Use the disparity map to warp the right image 
    for counter, img_pat in enumerate(left_validation_lst):
        # Read the images
        original_left_image = cv2.imread(left_validation_lst[counter])
        original_right_image = cv2.imread(right_validation_lst[counter])
        # Rescale the images
        left_image = cv2.resize(original_left_image,(620,188))
        input_left_image = left_image.reshape((1,188,620,3))
        input_left_image = input_left_image.astype(float)
        # Modify the input left image to the paper's specifications
        input_left_image = np.divide(np.subtract(input_left_image,128),255)
        # Down sample the left image to be the same as the right image
        rescaled_left_image = cv2.resize(original_left_image,(78,24))
        rescaled_left_image = rescaled_left_image.reshape((1,24,78,3))
        rescaled_left_image_tens = tf.constant(rescaled_left_image,dtype=tf.float32)
        rescaled_left_image = rescaled_left_image.astype(float)
        # Modify the input left image to the paper's specifications
        rescaled_left_image = np.divide(np.subtract(rescaled_left_image,128),255)
        # Down sample the left image to be the same as the right image
        rescaled_right_image = cv2.resize(original_right_image,(78,24))
        rescaled_right_image = rescaled_right_image.reshape((1,24,78,3))
        rescaled_right_image = rescaled_right_image.astype(float)
        # Modify the input left image to the paper's specifications
        rescaled_right_image = np.divide(np.subtract(rescaled_right_image,128),255)
        # Predict the disparity
        disparity = model.predict(input_left_image)
        # Calculate the horizontal gradient of the right, rescaled image
        h_grad = np.zeros((1,24,78,3))      
        h_grad[0,:,:,0] = cv2.Scharr(rescaled_right_image[0,:,:,0],cv2.CV_64F,1,0)
        h_grad[0,:,:,1] = cv2.Scharr(rescaled_right_image[0,:,:,1],cv2.CV_64F,1,0)
        h_grad[0,:,:,2] = cv2.Scharr(rescaled_right_image[0,:,:,2],cv2.CV_64F,1,0)       
        h_grad = np.float32(h_grad)
        # Set the previous image to be the rescaled right image
        prev_warped_image = rescaled_right_image
        # Generate the linearized warp
        warped_right_img = prev_warped_image + np.multiply(disparity,h_grad)
        # Calculate the photometric error
        photo_err = np.sum(np.square(np.subtract(rescaled_left_image,rescaled_right_image)))
        # Calculate the discontinuities in the disparity field
        disparity_diff = np.sum(np.square(disparity[0,:,:-1,0] - disparity[0,:,1:,0]))
        # Compute the loss
        validation_loss = validation_loss + photo_err + (0.01*disparity_diff)

        return validation_loss

 

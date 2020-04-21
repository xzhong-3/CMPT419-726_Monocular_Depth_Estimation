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


# Train model
train_path = "file path to training images (each rectified stereo pair should be contained in separate folders)."

# Construct a list to store the file paths to the individual images
left_img_pat_lst = []
right_img_pat_lst = []

for folders in os.listdir(train_path):
    # Construct a new file path to access the images in the folders
    new_path = train_path + "\\" + folders

    for files in os.listdir(new_path):
        if "left" in files:
           left_img_pat_lst.append(new_path + "\\" + files)
        elif "right" in files:
           right_img_pat_lst.append(new_path + "\\" + files)

# Remove 60% of the data for validation
validation_percentage = 0.6

left_validation = left_img_pat_lst[int(round(np.multiply(len(left_img_pat_lst),(1-validation_percentage)),0)):]
right_validation = right_img_pat_lst[int(round(np.multiply(len(right_img_pat_lst),(1-validation_percentage)),0)):]

# Construct the training list
left_training = left_img_pat_lst[0:int(round(np.multiply(len(left_img_pat_lst),(1-validation_percentage)),0))]
right_training = right_img_pat_lst[0:int(round(np.multiply(len(right_img_pat_lst),(1-validation_percentage)),0))]


def get_model():
    # Define the input
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
    conv6 = Conv2D(filters=2048, kernel_size=(5,5), activation='relu', strides=(1,1), padding='same',kernel_initializer = initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None),kernel_regularizer=l2(0.0005))(conv5_mp)
    # Output from layer 7 (should only be one image)
    conv7 = Conv2DTranspose(filters=1, kernel_size=(1,1), strides=(1, 1), padding='valid',  activation='relu', kernel_initializer=initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None),kernel_regularizer=l2(0.0005))(conv6)
    # Define the input and outputs to the model
    return Model(inputs = inputs, outputs = conv7)

# Instantiate a global counter
g_count = 0

for big_epoch in range (5):

    for img_counter in range (len(left_training)):

        original_left_image = cv2.imread(left_training[img_counter])
        original_right_image = cv2.imread(right_training[img_counter])

        left_image = cv2.resize(original_left_image,(620,188))
        input_left_image = left_image.reshape((1,188,620,3))
        input_left_image = input_left_image.astype(float)
        # Modify the input left image to the paper's specifications
        input_left_image = np.divide(np.subtract(input_left_image,128),255)
        input_left_image_tens = tf.convert_to_tensor(input_left_image,dtype=tf.float32)

        # Down sample the left image to be the same as the right image
        rescaled_left_image = cv2.resize(original_left_image,(20,6))
        rescaled_left_image = rescaled_left_image.reshape((1,6,20,3))
        rescaled_left_image_tens = tf.constant(rescaled_left_image,dtype=tf.float32)
        rescaled_left_image = rescaled_left_image.astype(float)
        # Modify the input left image to the paper's specifications
        rescaled_left_image = np.divide(np.subtract(rescaled_left_image,128),255)
        rescaled_left_image_tens = tf.constant(rescaled_left_image,dtype=tf.float32)

        # Down sample the left image to be the same as the right image
        rescaled_right_image = cv2.resize(original_right_image,(20,6))
        rescaled_right_image = rescaled_right_image.reshape((1,6,20,3))
        rescaled_right_image = rescaled_left_image.astype(float)
        # Modify the input left image to the paper's specifications
        rescaled_right_image = np.divide(np.subtract(rescaled_right_image,128),255)
        rescaled_right_image_tens = tf.constant(rescaled_right_image,dtype=tf.float32) 

        # Train on an individual image pair for 20 epochs
        for mini_epoch in range (1):

            # If this is the first iteration, instantiate the necessary variables
            if g_count == 0:
                # The model will be the vanilla unmodified
                model = get_model()
                # The previous disparity value will be set to zero
                prev_disparity = np.zeros((1,6,20,1))
                prev_disparity_tens = tf.constant(prev_disparity,dtype=tf.float32)
                h_grad = np.zeros((6,20,3))
                # Compute the horizontal gradient for the original right image
                h_grad[:,:,0] = cv2.Scharr(rescaled_right_image[0,:,:,0],cv2.CV_64F,1,0)
                h_grad[:,:,1] = cv2.Scharr(rescaled_right_image[0,:,:,1],cv2.CV_64F,1,0)
                h_grad[:,:,2] = cv2.Scharr(rescaled_right_image[0,:,:,2],cv2.CV_64F,1,0)
                # Change the data type of the array to float32
                h_grad = np.float32(h_grad)
                # Reshape the array and convert into tensor
                h_grad_tens = tf.convert_to_tensor(h_grad.reshape((1,6,20,3)),dtype=tf.float32)
                # The previous disparity will be set to zero with the new image
                prev_warped_image = rescaled_right_image
                prev_warped_image_tens = tf.convert_to_tensor(prev_warped_image,dtype=tf.float32)
                    

            elif mini_epoch == 0 and g_count !=0 :
                # If a new set of images received, start training from scratch again
                model = get_model()
                model.load_weights("L7_weights.h5") 
                # Given that there is no previous warped image. Compute all of the previous parameters based on the right image
                prev_disparity = np.zeros((1,6,20,1))
                prev_disparity_tens = tf.constant(prev_disparity,dtype=tf.float32)
                # Compute the horizontal gradient for the original right image
                h_grad[:,:,0] = cv2.Scharr(rescaled_right_image[0,:,:,0],cv2.CV_64F,1,0)
                h_grad[:,:,1] = cv2.Scharr(rescaled_right_image[0,:,:,1],cv2.CV_64F,1,0)
                h_grad[:,:,2] = cv2.Scharr(rescaled_right_image[0,:,:,2],cv2.CV_64F,1,0)
                # Change the data type of the array to float32
                h_grad = np.float32(h_grad)
                # Reshape the array and convert into tensor
                h_grad_tens = tf.convert_to_tensor(h_grad.reshape((1,6,20,3)),dtype=tf.float32)
                # The previous disparity will be set to zero with the new image
                prev_warped_image = rescaled_right_image
                prev_warped_image_tens = tf.convert_to_tensor(prev_warped_image,dtype=tf.float32)
        
            else:
                # The previously rescaled right image for the new set of data will be using the current right image and 
                # previous disparity value
                rep_current_disparity = np.tile(current_disparity,(1,1,3))
                rep_prev_disparity = np.tile(prev_disparity,(1,1,3))
                prev_warped_image = np.add(np.multiply(np.subtract(rep_current_disparity,rep_prev_disparity),h_grad),prev_warped_image)
                prev_warped_image_tens = tf.convert_to_tensor(prev_warped_image,dtype=tf.float32)
                # Set current parameters to previous parameters for subseqeunt training
                prev_disparity = current_disparity
                prev_disparity_tens = tf.constant(prev_disparity,dtype=tf.float32)
                # Compute the horizontal gradeint for previously warped image
                h_grad[:,:,0] = cv2.Scharr(prev_warped_image[0,:,:,0],cv2.CV_64F,1,0)
                h_grad[:,:,1] = cv2.Scharr(prev_warped_image[0,:,:,1],cv2.CV_64F,1,0)
                h_grad[:,:,2] = cv2.Scharr(prev_warped_image[0,:,:,2],cv2.CV_64F,1,0)
                # Change the data type of the array to float32
                h_grad = np.float32(h_grad)
                # Reshape the array and convert into tensor
                h_grad_tens = tf.convert_to_tensor(h_grad.reshape((1,6,20,3)),dtype=tf.float32)
                model = get_model()
                model.load_weights("L7_weights.h5")
        
            # Obtain a disparity prediction with the current weight settings
            current_disparity = model.predict(input_left_image)
            print(current_disparity)


            #Define the custom loss function
            def loss_fn(h_grad_tens,prev_warped_image_tens,rescaled_left_image):
                def loss(predicted_disparity,prev_disparity_tens):
                    # Increase the dimensionality of the the predicted and previous dispairity values
                    exp_predicted_disparity = K.repeat_elements(predicted_disparity,3,3)
                    exp_prev_disparity_tens = K.repeat_elements(prev_disparity_tens,3,3)
                    return (K.square((((exp_predicted_disparity-exp_prev_disparity_tens)*h_grad_tens)+prev_warped_image_tens)-rescaled_left_image) + (0.01*K.square(((K.concatenate((exp_predicted_disparity[:,:,1:,:],tf.constant(np.zeros((1,6,1,3)),dtype=tf.float32)),axis=2)-exp_predicted_disparity)))))/3  
                return loss

            # Defining the optimizer
            sgd = SGD(lr=(np.true_divide(np.true_divide(1,np.power(1+np.multiply(0.0005,big_epoch+1),(big_epoch))),100)), momentum=0.9, nesterov=False)
            print(np.true_divide(np.true_divide(1,np.power(1+np.multiply(0.0005,big_epoch+1),(big_epoch))),100))
            # Compile the model
            model.compile(loss = loss_fn(h_grad_tens,prev_warped_image_tens,rescaled_left_image), optimizer = sgd)

            # Perform back propagation with the current set of images again
            model.train_on_batch(x=input_left_image, y=prev_disparity)

            # Save the model
            model.save_weights("L7_weights.h5")

            # Increment the global counter
            g_count = g_count + 1
            print("Currently in global epoch {}".format(big_epoch))
            print("Currently on image {}".format(img_counter))

            # Destroy the previous session
            del model
            gc.collect()
            K.clear_session()
            tf.compat.v1.reset_default_graph()

        


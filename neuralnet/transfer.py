#Adapted from the tensorflow transfer learning tutorial:
#https://www.tensorflow.org/tutorials/images/transfer_learning

#conda install tensorflow-gpu

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #Removing this will result in much more introductory output
import tensorflow as tf
import time



# Lets define some variables for re-training this network.
EPOCHS      =20
IMG_SIZE    =(160,160)
IMG_SHAPE   = IMG_SIZE + (3,)#Im not sure what this does
ALPHA       = 0.0001#Learning rate


#Translation variables
angle_range = 20
scaling     = 1./127.5





# Now we can start processing our images for training on the network. We're applying translations
# with the parameters given above

#randomly flip horizontally, rotate up to 20 degrees, rescale and shift from [0,255] to [-1,1]
#There are many more transformations possible, look up the documentation.


print(f"Generating train/test data")
#Create Training
idg=tf.keras.preprocessing.image.ImageDataGenerator(
                                                    horizontal_flip=True,
                                                    rotation_range=20,
                                                    rescale=scaling,
                                                    samplewise_center=True)
train_dataset=idg.flow_from_directory('./images/training',target_size=IMG_SIZE)

#Create Testing
idg=tf.keras.preprocessing.image.ImageDataGenerator(
                                                    rescale=scaling,
                                                    samplewise_center=True)
validation_dataset=idg.flow_from_directory('./images/validation',target_size=IMG_SIZE)


print(f"Download pretrained model")

# Set some variables
NUM_CLASSES=len(validation_dataset.class_indices)


#And prepare the model
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, #Take off the output layers
                                               weights='imagenet')
base_model.trainable = False #Don't allow the backbone to train
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


print(f"Generating and train model")

#Lets prepare and run the model now
#Feed together image inputs, to the backbone, to a pooling layer, to our own layers of a
#Dropout later, a hidden layer of 10 neurons, then an output layer of NUM_CLASSES neurons.
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(10,activation='relu')(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES)(x)
model = tf.keras.Model(inputs, outputs)



#Actually build the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Train the data and get the model history
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset,verbose=2)

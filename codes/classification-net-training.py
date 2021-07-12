#!/usr/bin/env python
# coding: utf-8

import logging
import types

import os, fnmatch
import matplotlib.pyplot as plt 
import pandas as pd

import glob
import numpy as np

from random import seed
from random import randint

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from scipy.stats import poisson

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.metrics import categorical_accuracy

import pickle as pkl

#####################
#cnn model
#####################

def get_cnn():
    inputs = (Input((7,7,5)))
    net = BatchNormalization()(inputs)
    net = Conv2D(8, (3,3), activation='relu', padding="same")(net)
    net = Conv2D(16, (3,3), activation='relu', padding="same")(net)
    net = MaxPooling2D()(net)
    net = Conv2D(16, (2,2), activation='relu', padding="same")(net)
    net = Conv2D(32, (2,2), activation='relu', padding="same")(net)
    net = Dropout(0.1)(net)
    net = Flatten()(net)
    net = Dense(128, activation='relu')(net)
    net = Dropout(0.4)(net)
    net = Dense(64, activation='relu')(net)
    net = Dropout(0.2)(net)
    net = Dense(32, activation='relu')(net)
    net = Dropout(0.1)(net)
    net = Dense(16, activation='relu')(net)
    outputs = Dense(3, activation='softmax')(net)
    model = Model(inputs=[inputs], outputs=[outputs])

    model.summary()
    
    return model

def get_samples(ims):
    return (poisson.rvs(ims) - norm_mean) / norm_std

def generator_train(patches_list, batch_size):

    number_of_batches = len(patches_list)//batch_size
    #np.random.shuffle(patches_list)

    patches_list = patches_list.sample(frac=1, random_state=random_seed)
    counter=0
    
    while 1:
        
        if counter >= number_of_batches:
            #np.random.shuffle(patches_list)
            patches_list = patches_list.sample(frac=1, random_state=random_seed)
            counter = 0
        
        ini_for = min(batch_size*counter, len(patches_list))
        end_for = min(batch_size*(counter+1), len(patches_list))

        batch_size_local = end_for-ini_for

        if (batch_size_local > 0):
            x_batch = np.zeros((batch_size_local, xsize, xsize, 5), dtype=float)
            y_batch = np.zeros((batch_size_local, 3), dtype=float)
        
            for con in range(batch_size_local):

                #image counter starts with ini_for
                index_image = ini_for + con
                
                image_file = patches_list["filename"].iloc[index_image]
                image_type = patches_list["class"].iloc[index_image]

                #read the asimov data image
                x_a = np.load(f'{path_to_data}/{image_file}')
                
                #create the one-hot encoder
                y_a = np.array([0,0,0])
                y_a[int(image_type)]=1
                
                #generate the batch elements
                x_batch[con,:,:,:] = x_a[:,:,:]*global_correction_factor
                y_batch[con,:] = y_a

            #don't forget to from scipy.stats import poisson
            yield get_samples(x_batch), y_batch

        counter += 1

def generator_valid(patches_list, batch_size):

    number_of_batches = len(patches_list)//batch_size
    patches_list = patches_list.sample(frac=1, random_state=random_seed) 
    counter=0
    
    while 1:
        
        if counter >= number_of_batches:
            #np.random.shuffle(patches_list)
            patches_list = patches_list.sample(frac=1, random_state=random_seed)
            counter = 0
        
        ini_for = min(batch_size*counter, len(patches_list))
        end_for = min(batch_size*(counter+1), len(patches_list))

        batch_size_local = end_for-ini_for

        if (batch_size_local > 0):
            x_batch = np.zeros((batch_size_local, xsize, xsize, xdepth), dtype=float)
            y_batch = np.zeros((batch_size_local, ydepth), dtype=float)
        
            for con in range(batch_size_local):

                #image counter starts with ini_for
                index_image = ini_for + con

                image_file = patches_list["filename"].iloc[index_image]
                image_type = patches_list["class"].iloc[index_image]

                #read the asimov data image
                x_a = np.load(f'{path_to_data}/{image_file}')
                
                #create the one-hot encoder
                y_a = np.array([0,0,0])
                y_a[int(image_type)]=1
                
                #generate the batch elements
                x_batch[con,:,:,:] = x_a[:,:,:]*global_correction_factor 
                y_batch[con,:] = y_a[:]

            #don't forget to from scipy.stats import poisson
            yield get_samples(x_batch), y_batch

        counter += 1

##########################################
#solid angle correction factor
##########################################
#global_correction_factor = 0.12
global_correction_factor = 1.0
##########################################
        
#epochs
epochs = 500

xsize = 7
ysize = 7

xdepth = 5
ydepth = 3

batch_size = 128

#let us fix the np random seed
#np.random.seed(23)
random_seed = 23

#patches_classification_training and csv_files folders can be downloaded from zenodo repo
#https://zenodo.org/record/4587205#.YFOjKv7Q9uR
path_to_data = "../patches_classification_training"
path_to_csv = "../csv_files"

path_to_model = "../models"
pickle_file = "../models/standard_norm_vals.pkl"

##########################
#data load, csv lists
##########################
data_train = pd.read_csv(f'{path_to_csv}/training_classification.csv')
data_valid = pd.read_csv(f'{path_to_csv}/validation_classification.csv') 

#length of patches list
len_train = len(data_train)
len_valid = len(data_valid)

print("train and valid lengths: ",len_train, len_valid)

#########################
#load mean and std array
#########################

with open(pickle_file,"rb") as f:
    [norm_mean, norm_std] = pkl.load(f)

#############################
##### train and validation
#############################

model = get_cnn()

#model compile
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[categorical_accuracy])

earlystopper = EarlyStopping(monitor='val_loss', patience=50, verbose=1, min_delta=1e-7)

checkpointer = ModelCheckpoint(f"{path_to_model}/cnn_model.h5", monitor='val_loss',\
                               verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=0.000000001)

model.fit_generator(
    generator_train(data_train, batch_size),
    epochs=epochs,
    steps_per_epoch = len_train//batch_size,
    validation_data = generator_valid(data_valid, batch_size*2),
    validation_steps = len_valid//batch_size*2,
    verbose=1,
    callbacks=[checkpointer, earlystopper, reduce_lr]
)







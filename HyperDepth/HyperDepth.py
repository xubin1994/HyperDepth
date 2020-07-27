import cv2
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import os

import utilities as utils
import evaluations as evals

#this file contains the primary code for the HyperDepth implementation
#CURRENTLY CONTAINS:
# --> RFC setup & training

#path to GT data [fill in later]
data_path = "\data_path"
labels_path = "\labels_path"

# for now assuming image resolution of 1920x1280 -- may change later
num_pixels_x = 1280 # length of scanline
num_pixels_y = 1920
num_features = 64 # might change
forest_depth = 12
train_test_split_frac = 0.8
num_trees = 4 # arbitrary, will change
pixel_radius = 15 # pixel radius for rand features

#calculate random number of line displacements
displacement_idx = utils.generate_displacements(num_features, pixel_radius)

#get list of image filenames
imgList = utils.list_images(data_path)

#grab samples from imgList
imgList = imgList.sample(n=100) #n arbitrary for now

#reset indicies to reflect extracted samples
imgList = imgList.reset_index(drop = True)


#extract features from images

#set up RFC

#loop through scanlines & train model
line_idxs = np.arange(0, 1280) # indices of image to loop through -- WILL CHANGE
models = []

# extract feature vecs from images

# set up models
for x in range(len(line_idxs)):
    models.append(RandomForestClassifier(n_estimators = num_trees, max_features = 1, max_depth = forest_depth, criterion = 'entropy', 
                                         random_state = 1234, verbose = 0, n_jobs = 2))

# set up temp vars
train_acc = [0]*len(line_idxs)
test_acc = [0]*len(line_idxs)
train_rmse = [0]*len(line_idxs)
test_rmse = [0]*len(line_idxs)
kept_feats = [0]*len(line_idxs)

# loop through scanlines
for idx in range(len(line_idxs)):
    print("Calculating line " + idx + "of " + len(line_idxs))

    # train-test split

    # fit model

    # calculate accuracy

















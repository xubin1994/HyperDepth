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
num_pixels_x = 1280
num_pixels_y = 1920
num_features = 64 # might change
forest_depth = 8
pixel_radius = 15 # pixel radius for rand features

#calculate random number of line displacements
displacement_idx = utils.generate_displacements(num_features, pixel_radius)

#get list of image filenames
imgList = utils.list_images(data_path)

#grab samples from imgList
imgList = imgList.sample(n=100) #n arbitrary for now

#reset indicies to reflect samples
imgList = imgList.reset_index(drop = True)


#extract features from images

#set up RFC

#loop through scanlines & train model















import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# this file contains methods for evaluating model predictions


def accuracy(labels, logits):
    return np.mean(labels==logits)

def mse(labels, logits):
    return np.mean(np.square(np.float64(labels) - np.float64(logits)))

def rmse(labels, logits):
    return (sqrt(mse(labels, logits)))


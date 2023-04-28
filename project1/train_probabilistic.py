# -*- coding: utf-8 -*-
"""
ESE650-Learning in Robotics Project 1

@author: Achille Verheye

This script:
- loads in training images from a folder
- presents the images to the user who selects regions that satisfy the
specified color or no region if not present
- convert the images to a color space that separates luma from chroma
currently using the HSV color space
- trains a probabilistic model (Gaussian or Gaussian Mixture) for the specified
color
- saves model details to file
"""

# temporary warning ignoring:
# analysis:ignore

import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

from polygon import create_polygon_mask

# specify the color the user will select. This name is used in the saved model
# file
color = 'barrel_red'

# initialize list that holds mus and sigmas for each image
mu_list = []
sigma_list = []

# load in the training images
training_folder = "D:/Penn/2/ESE650-LR/project1/Proj1_Train_alternative/train_t"
for filename in os.listdir(training_folder):
    # read in image
    img = cv2.imread(os.path.join(training_folder, filename))
    # convert color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # show image
    fig, (ax1) = plt.subplots()
    ax1.imshow(img_HSV)
    
    xy = plt.ginput(4)
    xy_np = np.array(xy)
    
    #close the figure
    plt.close()
#    xpoints, ypoints = zip(*xy)
#    ax1.fill(xpoints, ypoints, color='lightblue')
#    ax1.plot(xpoints, ypoints, ls='--', mfc='red', marker='o')
#    plt.show()
    
    # create a mask from these vertices
    mask = create_polygon_mask(shape=img_HSV.shape[0:2], vertices=xy_np)
    # create a masked array to isolate the color fields
    masked_img_H = np.ma.masked_array(img_HSV[:,:,0], mask=mask)
    masked_img_S = np.ma.masked_array(img_HSV[:,:,1], mask=mask)
    masked_img_V = np.ma.masked_array(img_HSV[:,:,2], mask=mask)
    
    #NOTE: looks like the first channel, H, yields good segmentation of the barrel
#    fig,(ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,4))
#    ax1.imshow(masked_img_H)
#    ax2.imshow(masked_img_S)
#    ax3.imshow(masked_img_V)

    # now that we have the subset of the image containing the desired colored
    # pixels, we can use MLE to determine the most likely parameters mu, sigma
    # for the Gaussian distribution representing this color class
    # mu has a dimension of length 3, one element for each color component(HSV)
    mu = np.array([[np.mean(masked_img_H)], [np.mean(masked_img_S)], [np.mean(
            masked_img_V)]])
    mu_list.append(mu)
#    xminmu_H = np.expand_dims((masked_img_H.compressed() - mu[0]), axis=1)
#    unnorm_cov_H = np.dot(xminmu_H, xminmu_H.T)
#    cov_H = unnorm_cov_H/unnorm_cov_H.size
#    
#    xminmu_S = np.expand_dims((masked_img_S.compressed() - mu[1]), axis=1)
#    unnorm_cov_S = np.dot(xminmu_S, xminmu_S.T)
#    cov_S = unnorm_cov_S/unnorm_cov_S.size
#    
#    xminmu_V = np.expand_dims((masked_img_V.compressed() - mu[2]), axis=1)
#    unnorm_cov_V = np.dot(xminmu_V, xminmu_V.T)
#    cov_V = unnorm_cov_V/unnorm_cov_V.size
    
    sigma = np.array([[np.cov(masked_img_H.compressed())],
                      [np.cov(masked_img_S.compressed())],
                            [np.cov(masked_img_V.compressed())]])
    sigma_list.append(sigma)
    
    
    
    

    
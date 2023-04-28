# -*- coding: utf-8 -*-
"""
Script for testing the localization and distances of the red barrels

@author: Achille Verheye
"""
import os

from red_barrel_detector import RedBarrelDetector

# training and testing parameters
#################################
# folder that contains the training images
training_folder = "D:/Penn/2/ESE650-LR/project1/Proj1_Train_alternative/training"

########################## SET THIS VARIABLE ##################################
testing_folder = "D:/Penn/2/ESE650-LR/project1/testset" 
###############################################################################

# set this to True if you want to train the model. If you want to test new
# images, set to False
training_mode = False
train_distances = False
# colors that you want to train if in training mode. Colors that already have
# been modeled don't have to be retrained. Colors that are already present
# will be overwritten
training_colors = ['barrel_red', 'brown', 'other_red']
# specify the number of Gaussian mixtures
nb_mixtures = 2

if training_mode:
    # create RBD model
    rbd_train = RedBarrelDetector(nb_mixtures = nb_mixtures)
    # train model for each color
    for color in training_colors:
        rbd_train.train_color_model(training_folder, color)
        
elif train_distances:
    # create RBD model
    rbd_train = RedBarrelDetector(nb_mixtures = nb_mixtures)
    # train distances
    rbd_train.train_distances(training_folder, theta_filename ="theta.npy")
    
else:    
    for filename in os.listdir(testing_folder):
        # create RBD model
        rbd = RedBarrelDetector(img_path=os.path.join(testing_folder,filename))
        # find red pixels
        rbd.find_red_pixels()
        # find barrels
        rbd.find_barrels()
        # plot the result
        rbd.plot_result()
        # plot bounding box results
        print(rbd.img_path, rbd.boundingboxes)

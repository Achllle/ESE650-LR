# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:48:49 2017

@author: Achille
"""
import numpy as np
from skimage.draw import polygon

def create_polygon_mask(shape, vertices):
    """Create a 2D from 2D vertices"""
    xpoints, ypoints = zip(*vertices)
    # swap x and y because polygon from scikit-image has diff coord system
    poly = polygon(ypoints, xpoints, shape)
    mask = np.ones(shape=shape)
    mask[poly] = 0
    
    return mask
# -*- coding: utf-8 -*-
"""
Class file that contains the algorithm for detecting red barrels in images

@author: Achille Verheye
"""

import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy

from polygon import create_polygon_mask
from linreg_closedform import LinearRegressionClosedForm as linreg

class RedBarrelDetector(object):
    """
    A red barrel detector
    """
    
    def __init__(self, img_path = None, nb_mixtures=1):
        """
        Initialize this red barrel detector.
        """
        # use the existing color models if available
        self.color_models_filename = "color_models.npy"
        self.theta_filename = "theta.npy"
        self.img_path = img_path
        self.nb_mixtures = nb_mixtures
        if img_path is not None:
            self.load_original_image(img_path)
            self.load_image()
        try:
            self.colormodels = np.load(self.color_models_filename, allow_pickle=True)[()]
        except FileNotFoundError:
            self.colormodels = None
        try:
            self.dist_theta = np.load(self.theta_filename)[()]
        except FileNotFoundError:
            self.theta_filename = None
    
    def load_image(self):
        """
        Convert loaded image to the appropriate color space
        
        A numpy array of size (rows, cols, 3), in the H S V color space
        """
        # convert color space to HSV
        self.img = cv2.cvtColor(self.origimg, cv2.COLOR_BGR2HSV)
    
    def load_original_image(self, img_path):
        """
        Load in an image using the given path
        """
        self.origimg = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
    def train_color_model(self, training_folder, color):
        """
        Train a color model for a given color.
        
        - load in training images from a folder
        - present the images to the user who selects regions that satisfy the
        specified color or no region if not present
        - convert the images to a color space that separates luma from chroma
        currently using the HSV color space
        - train a probabilistic model (Gaussian or Gaussian Mixture) for the
        specified color
        - save model details to file
        
        @param  training_folder
                String containing the location of the images that are used to 
                be trained the color model
        """
        # initialize list that holds mu's and sigma's for each image
        mu_list_H = np.array([])
        mu_list_S = np.array([])
        mu_list_V = np.array([])
        sigma_list_H = np.array([])
        sigma_list_S = np.array([])
        sigma_list_V = np.array([])
        
        for filename in os.listdir(training_folder):
            # load in image
            img_path = os.path.join(training_folder,filename)
            self.load_original_image(img_path)
            self.load_image()
            
            # show image
            fig, (ax1) = plt.subplots()
            ax1.imshow(self.origimg)
            plt.title('select the area that has color {}. Enter to skip'
                      .format(color))
            dont_skip = True
            try:
                xy = plt.ginput(4)
                if len(xy) != 4:
                    raise ValueError
                xy_np = np.array(xy)
            except ValueError: # pressed enter, no pixels with that color
                dont_skip = False
            
            #close the figure
            plt.close()
            
            if dont_skip: # didn't press enter
                # create a mask from these vertices
                mask = create_polygon_mask(shape=self.img.shape[0:2], vertices=xy_np)
                # create a masked array to isolate the color fields
                masked_img_H = np.ma.masked_array(self.img[:,:,0], mask=mask)
                masked_img_S = np.ma.masked_array(self.img[:,:,1], mask=mask)
                masked_img_V = np.ma.masked_array(self.img[:,:,2], mask=mask)
            
                # if self.nb_mixtures == 1:
    
                # now that we have the subset of the image containing the desired colored
                # pixels, we can use MLE to determine the most likely parameters mu, sigma
                # for the Gaussian distribution representing this color class
                # mu has a dimension of length 3, one element for each color component(HSV)
                mu_H = masked_img_H.compressed()
                mu_S = masked_img_S.compressed()
                mu_V = masked_img_V.compressed()
                mu_list_H = np.hstack((mu_list_H, mu_H))
                mu_list_S = np.hstack((mu_list_S, mu_S))
                mu_list_V = np.hstack((mu_list_V, mu_V))
                
                sigma_H = masked_img_H.compressed()
                sigma_S = masked_img_S.compressed()
                sigma_V = masked_img_V.compressed()
                sigma_list_H = np.hstack((sigma_list_H, sigma_H))
                sigma_list_S = np.hstack((sigma_list_S, sigma_S))
                sigma_list_V = np.hstack((sigma_list_V, sigma_V))
        
        model_mu = np.array([np.mean(mu_list_H), np.mean(mu_list_S),
                             np.mean(mu_list_V)])
        combined_sigma = np.vstack((sigma_list_H, sigma_list_S, sigma_list_V))
        model_sigma = np.cov(combined_sigma)
        
        model = {'mu':model_mu,'sigma':model_sigma}
        # save the color model to file
        self.store_color_model(color, model)
        
    def store_color_model(self, color, model):
        """
        Add the color model to the color model file
        
        If no color model exists, create the file. Loading the file returns
        a numpy array containing a dictionary mapping the colors (strings) to
        their respective models
        """          
        try:
            self.colormodels[color] = model
        except TypeError:
            self.colormodels = {color:model}
        
        np.save(self.color_models_filename, self.colormodels)
    
    def train_distances(self, training_folder, theta_filename):
        """
        Train a model that relates pixel width of an image to distance to the
        barrel
        """
        actual_distances = []
        sizes = []
        
        for filename in os.listdir(training_folder):
            # create RBD model
            rbd = RedBarrelDetector(img_path=os.path.join(training_folder,filename))
            # find red pixels
            rbd.find_red_pixels()
            # find barrels
            rbd.find_barrels()
            
            # load in distance from filename
            dist = filename.split('.')[0]
            if '_' in dist:
                # skip this case, not worth the bookkeeping effort...
                continue
            
            try:
                box = rbd.boundingboxes[0]
            except (IndexError, AttributeError):
                continue
            # find euclidean distances
            dist1 = np.linalg.norm(box[0]-box[1])
            dist2 = np.linalg.norm(box[1]-box[2])
            width = min([dist1, dist2])
            height = max([dist1, dist2])
            area = width*height
            
            actual_distances.append(dist)
            sizes.append(area)
            
        # now train a linear classifier on the distances vs areas
        X = np.expand_dims(np.array(sizes),axis=1)
        y = np.expand_dims(np.array(actual_distances),axis=1).astype('float64')
        linear_model = linreg()
        linear_model.fit(X, y)
        # save to file
        np.save(theta_filename, linear_model.theta)
        
    
    ###########################################################################
    # methods related to red barrel detection for a single image
    ###########################################################################
    def find_red_pixels(self):
        """
        Segment image into multiple colors based on their similarity to the
        color classes as defined by the Gaussian models.
        Segment the given image into multiple segments using the color models
        Each pixel is assigned a color from one of the color models based
        on its Mahalanobis distance to the models. The pixels that are closest
        to the model of the red barrel are returned as a binary mask
        
        @store  red_barrel_pix
                A 2D binary numpy array containing True for pixels that are 
                considered red barrel color.
        """
        # initialize distance matrix 
        distance_matrix_index_to_color_mapping = {}
        index = 0
        distances_matrix = []
        for color, model in self.colormodels.items():
            distance_matrix_index_to_color_mapping[color] = index
            index += 1                                                  
            distance_to_color = self.compute_model_dist(color, model)
            distances_matrix.append(distance_to_color)
        
        segmented = np.argmax(np.array(distances_matrix), axis=0)
        index_barrel_red = distance_matrix_index_to_color_mapping['barrel_red']
        
        red_barrel_pix = np.where(segmented == index_barrel_red, 1, 0)
        self.red_barrel_pix = red_barrel_pix.astype('bool_')

        
    def compute_model_dist(self, color, model):
        """
        Compute the distances for each pixel from an image to a model
        
        @return distances
                numpy array of the same shape of the image but width depth 1
                that contains the Mahalanobis distance from each pixel to the
                specified color model
        """
        # extract mu and sigma from the model
        mu = model['mu']
        sigma = model['sigma']
        
#        # unroll image to allow matrix operations
        rows, cols = self.img.shape[0:2]
#        x = np.reshape(self.img, (3, rows*cols))
        
        # calculate the likelihood, a pdf, to use in the Bayes equation
        # using the equation for the pdf of a Gaussian distribution
        distances = np.zeros((rows,cols))
                
        matrix = (self.img - mu).dot(scipy.linalg.fractional_matrix_power(sigma, -1/2))
        # square elementwise
        squared = np.square(matrix)
        # compute sum along 2nd dimension
        g = np.sum(squared, axis=2)
        # exponentiate
        distances = np.exp(-0.5*g)
#        pdf_denom = np.sqrt(((2*np.pi)**3)*np.linalg.det(sigma))
                
#        distances[row,col] = pdf_num - np.log(pdf_denom)
#        distances = pdf_num/pdf_denom
        
        return distances
        
    def find_barrels(self):
        """
        Locate the barrel(s) in the image (if any) and save the coordinates
        of the bounding boxes
        
        @store  barrels
                numpy array containing zero or more numpy arrays that represent
                masks of where barrels are located
        """
        # make a copy of self.red_barrel_pix as findContours modifies the
        # original image EDIT not anymore since OpenCV 3.2
        binary_img = self.red_barrel_pix.copy().astype('uint8')
        # connect components that are close?
        # closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel=kernel)
        # opening
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel=kernel)
        # label the image to find how many potential barrels there are
        
        # use cv2.CHAIN_APPROX_NONE for a closed region. Simple leaves out the
        # points that are along a straight line and thus saves memory
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        
        # remove tiny blobs: based on size relative to largest blob
        # find largest contour
        areas = []
        for contour in contours:
            # calculate area, append to areas list
            areas.append(cv2.contourArea(contour))
        try:
            largest_area = np.max(areas)
        except ValueError:
            self.boundingboxes = []
            return
        # remove items from contours that are too small
        # too small is defined as a area that is smaller than a fraction of the
        # largest area
        to_remove = []
        for ind, area in enumerate(areas):
            fraction = 0.02
            if area < fraction*largest_area:
                to_remove.append(ind)
        for index in sorted(to_remove, reverse=True):
            del contours[index]
        
        # initialize barrels and bounding boxes
        self.barrels = np.zeros(opening.shape)
        self.boundingboxes = []
        # initialize list of contours
        new_contours = []
        
        for index, contour in enumerate(contours):
            
            # calculate moments
            M = cv2.moments(contour)
            # from moments, calculate centroid
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            area = M['m00']
            
            # create image that holds this contour and has 'room' for contours
            # that are close.
            mask = np.zeros(opening.shape, np.uint8)
            cv2.drawContours(mask,[contour],0,255,-1) # -1 fills the shape
            
            # remaining contours
            rem_contours = contours[index+1:]
            for ind_rem, rem_contour in enumerate(rem_contours):
                # calculate moments
                M_rem = cv2.moments(rem_contour)
                # from moments, calculate centroid
                cx_rem = int(M_rem['m10']/M_rem['m00'])
                cy_rem = int(M_rem['m01']/M_rem['m00'])
                area_rem = M_rem['m00']
                # calculate distance between two contours
                dist_centroids = np.sqrt((cx-cx_rem)**2 + (cy-cy_rem)**2)
                # weighted distance
                weighted_dist = (dist_centroids**3)/(area_rem + area)
                
                # if distance is close, add that blob to mask
                if weighted_dist < 300:  
                    # add contour to image
                    cv2.drawContours(mask,[rem_contour],0,255,-1)
                    
            test_mask = np.zeros(opening.shape)
            # find contours in mask image so we can find the convex hull
            mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            # combine the contours to a single array
            single_contour = mask_contours[0]
            for mask_contour in mask_contours[1:]:
                single_contour = np.vstack((single_contour, mask_contour))
            # get the convex hull, result is a contour
            hull = cv2.convexHull(single_contour)
            # add to the list of contours
            new_contours.append(hull)
            cv2.drawContours(test_mask,[hull],0,1,-1)
            
        # get rid of overlapping contours
        all_barrels = np.zeros(opening.shape)
        for contour in new_contours:
            cv2.drawContours(all_barrels,[contour],0,1,-1)
        all_barrels = all_barrels.astype('uint8')
        final_contours, _ = cv2.findContours(all_barrels, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)                

        # after combining, check contours for size
        for contour in final_contours:
            # check contour size
            area = cv2.contourArea(contour)
            
            barrel = np.zeros(opening.shape)
            cv2.drawContours(barrel,[contour],0,1,-1)
            
            if True: # replace condition with sth else
                self.barrels = np.logical_or(self.barrels, barrel)
                # find bounding box for this contour
                box = self.find_bounding_boxes(contour)
                self.boundingboxes.append(box)       
        
    def plot(self, img):
        """Simple plotting method"""
        plt.figure()
        plt.imshow(img)
        
    def find_bounding_boxes(self, contour):
        """
        Find the coordinates of the bounding boxes if any
        
        @store  boundingboxes
                box
        """
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        return box
        
    def calculate_distance(self):
        """
        Calculate the distance from the camera to the barrels in meters
        
        This method expects box to contain four points in sequential order
        The rightmost point (highest column value) is the first (0) point, the
        next points are counter-clockwise.
        """
        distances = []
        # use projection matrix and bounding box
        for box in self.boundingboxes:
            # find euclidean distances
            dist1 = np.linalg.norm(box[0]-box[1])
            dist2 = np.linalg.norm(box[1]-box[2])
            width = min([dist1, dist2])
            height = max([dist1, dist2])
            area = width*height
            
            # create linreg model
            model = linreg()
            model.theta = self.dist_theta
            
            calc_dist = model.predict(np.array([area]))[0][0]
            # round
            calc_dist = np.round(calc_dist, decimals = 2)
        
            distances.append(calc_dist)
        
        return distances
        
    def plot_result(self):
        """
        Plot the result of the barrel finding algorithm
        Also add the calculated distance to the figure
        """
        fig, axarr = plt.subplots(2,2)
        
        # plot original image
        axarr[0,0].imshow(self.origimg)
        axarr[0,0].title.set_text('original image')
        
        # plot red_barrel segmented image
        axarr[0,1].imshow(self.red_barrel_pix)
        axarr[0,1].title.set_text('result from finding red barrel pixels')
        
        # plot barrel(s)
        axarr[1,0].imshow(self.barrels)
        axarr[1,0].title.set_text('result from finding red barrels')
        
        # plot original image along with bounding box
        bounding_and_orig = self.origimg.copy()
        for box in self.boundingboxes:
            cv2.drawContours(bounding_and_orig,[box],0,(255,0,0),3)
        axarr[1,1].imshow(bounding_and_orig)
        axarr[1,1].title.set_text('final result')

        # add distance as text
        dist = self.calculate_distance()
        # main title
        if len(dist) == 1:
            plt.suptitle('''Red Barrel Detection by Achille Verheye.
            Distance to red barrel is about {} meters'''.format("{0:.2f}".format(dist[0])))
        else:
            plt.suptitle('''Red Barrel Detection by Achille Verheye.
            Distances to red barrels are about {} meters
            '''.format(str(dist)))
        plt.show()
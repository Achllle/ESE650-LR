Instructions for using the red barrel localization files
Achille Verheye
--------------------------------------------------------
The zip file should contain the following files:
- red_barrel_detector.py
- rest_red_barrel.py
- polygon.py
- color_models.npy
- linreg_closedform.py
These files should all be in the same directory
--------------------------------------------------------
To test the model on a set of images:
- open test_red_barrel.py
- set the testing_folder variable to the path that contains the testing images (not the training_folder!)
- make sure training_mode and training_distance are set to False
- run the script
--------------------------------------------------------
To train the model
- open test_red_barrel.py
- set the training folder variable to the path that contains the training images
- set training_mode and training_distance to True
- to the variable 'training_colors', add colors that you want to train on
- run the script

Questions: averheye@seas.upenn.edu
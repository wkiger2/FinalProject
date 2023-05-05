# CS 598 Deep Learning for Healthcare - Spring Semester 2023
Final Project William Kiger (wkiger2@illinois.edu) and Kristopher Gallagher (kmg8@illinois.edu)

# A Second Look at "Fair Conformal Predictors for Applications in Medical Imaging" 
# From: https://arxiv.org/pdf/2109.04392v3.pdf
Steps to run our programs: 
1) The dataset and model must be downloaded from here: https://drive.google.com/drive/folders/1yC46cY7UR49-FMBja1Z_AEohmLnXuEWj?usp=sharing
2) Download the model "resnet_18_derm_model_test_v2.pt" and place it in the directory named "model"
3) Unzip the datasets "validation_dataset" & "data_train_valid_only" and place them in the directory named "data"
 The relative paths should look like this when finished: <br /> 
 FinalProject\data\data_train_valid_only\data which contains train and valid directories <br /> 
 FinalProject\data\validation_dataset\val which contains the 10 labeled directories with images <br /> 
 Please ensure your paths look like this before running our programs. <br /> 

4) Create conda environment <br /> 
There are 2 .yml files included in the repo to reproduce our conda environments, environment_nobuilds.yml which is less system dependant.  The complete .yml file with the build information is named environment.yml.  You may need to update your path in the prefix sections in the yml files. 

##################################################################
# Our testing of the RAPS conformal prediction algorithm references
# and reuses functions from the below paper and code repo.  
# 
# https://github.com/aangelopoulos/conformal_classification
# https://arxiv.org/abs/2009.14193
##################################################################

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
# import matplotlib.pyplot as plt
import time
from utils import *
from conformal import ConformalModel
import torch.backends.cudnn as cudnn
import random
import os


if __name__ == "__main__":

    np.random.seed(seed=0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)

    # Transform as in https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Get the conformal calibration dataset
    root = r'../data/validation_dataset' 
    validate_path = os.path.join(root, 'val')

    ##################################################################
    # WDK Code
    # Have to have the number of images in the calibration set
    # this is my check
    ##################################################################
    total_data = torchvision.datasets.ImageFolder(root)
    data_loader = torch.utils.data.DataLoader(total_data,
                                          batch_size=10,
                                          shuffle=True,)
    # print(len(total_data)) #  --> have to plug this into the line below
    
    calib_data, val_data = torch.utils.data.random_split(
        torchvision.datasets.ImageFolder(validate_path, transform), [100, len(total_data)-100]) #[num, num_image-num]
    ##################################################################
    # WDK Code End
    # now loading the data with the correct parameters
    ##################################################################

    # Initialize loaders
    calib_loader = torch.utils.data.DataLoader(
        calib_data, batch_size=10, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=10, shuffle=True, pin_memory=True)

    cudnn.benchmark = True

    ##################################################################
    # Check for device type and load the model
    ##################################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = r"../model/resnet_18_derm_model_test_v2.pt"
    model = torch.load(model_path)
    # model.to(device)

    model = torch.nn.DataParallel(model)  # wdk...for multi-gpu instances
    model.eval()

    # set more parmaeters
    # optimize for 'size' or 'adaptiveness'
    lamda_criterion = 'size'
    # allow sets of size zero
    allow_zero_sets = False
    # use the randomized version of conformal
    randomized = True

    # Conformalize model
    #TODSO:  IN THE UTILS FILE, THERE IS A FUNCITON get_logits_targets THAT HAS 1000IMAGENET CLASSES HARDCODED...HAVE TO CHANGE BACK!! 
    model = ConformalModel(model, calib_loader, alpha=0.1, lamda=0,
                           randomized=randomized, allow_zero_sets=allow_zero_sets)

    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    validate(val_loader, model, print_bool=True)

    print("Complete!")

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:01:17 2023

@author: WilliamKiger
"""
import os
import random
import numpy as np

#PyTorch imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

#local files
import naive_conformal_prediction.ncp_inference as ncpi
import naive_conformal_prediction.inference as inference


class NaiveConformalModel(nn.Module):
    def __init__(self, model, calib_loader, alpha):
        super(NaiveConformalModel, self).__init__()
        self.model = model 
        self.alpha = alpha
        self.calibration_thresholds = ncpi.calibrate(model, calib_loader, alpha)
        self.qhat = []


if __name__ == "__main__":
    
    #load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device = " + str(device))
    
    model_path = r"../model/resnet_18_derm_model_test_v2.pt"
    model = torch.load(model_path)
    # model.to(device)
    model = torch.nn.DataParallel(model)  # wdk...for multi-gpu instances
    #model to eval mode
    _ = model.eval()
    
    #calibration dataset prepared  --> have to pass path
    root = r'../data/validation_dataset'
    validate_path = os.path.join(root, 'val')  
    
    cmodel = NaiveConformalModel(model, validate_path, alpha=0.1)
    
    test_img = r'../data\validation_dataset\val\folliculitis\258.jpg'
    prediction_set = inference.NaiveCPredict(cmodel, test_img, print_bool=False)
    print("\nNaive Prediction Set:")
    print(prediction_set)
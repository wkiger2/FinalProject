# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 19:57:06 2023

@author: WilliamKiger
"""

from PIL import Image
import matplotlib.pyplot as plt 

#PyTorch imports
import torch
import torchvision.transforms as transforms
import naive_conformal_prediction.class_names as class_names


def NaiveCPredict(model, test_image_path, print_bool):
    cmodel = model.model
    prediction_set = []
    
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std= [0.229, 0.224, 0.225])
            ])

    test_image = Image.open(test_image_path)
    
    if print_bool: 
        plt.imshow(test_image)
    
    test_image_tensor = transform(test_image)
    
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    with torch.no_grad():
        cmodel.eval()
        # Model outputs log probabilities
        out = cmodel(test_image_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(10, dim=1)

        # cls = class_names[topclass[0][0]]
        # cls = class_names.class_name_lookup([topclass[0][0]]) #assigned but never used
        score = topk.cpu().numpy()[0][0]
        
        if print_bool: 
            print("Softmax Scores")      

        for i in range(10): #10 is the hardcoded num of classes
            topclass.cpu().numpy()[0][i]
            
            if print_bool: 
                print("Prediction", i+1, ":", class_names.class_name_lookup([topclass.cpu().numpy()[0][i]]), ", Score: ", topk.cpu().numpy()[0][i])
            # class_num = class_names[topclass.cpu().numpy()[0][i]]
            class_num = class_names.class_name_lookup([topclass.cpu().numpy()[0][i]])
            score = topk.cpu().numpy()[0][i]
            prediction_set.append((class_num, score))
            
    ncp_list = []
    
    #steps -- look up threshold 
    top_class = class_names.class_name_lookup([topclass[0][0]])
    threshold = [ (x,y) for x, y in model.calibration_thresholds if x  == top_class ]
    threshold = threshold[0][1]

    #append to new list everything greater than threshold 
    idx = 0 
    for i in prediction_set: 
        #first, divide by 100 to get true percentage
        label = prediction_set[idx][0]
        value = prediction_set[idx][1]
        
        if value >= threshold: 
            ncp_list.append((label,value))
        idx += 1
                 
    return ncp_list
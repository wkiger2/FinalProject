# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:28:12 2023

@author: WilliamKiger
"""
import numpy as np

#PyTorch imports
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import naive_conformal_prediction.class_names

#this will hold our softmax buckets for each class
class Cp_Class(list):
    def __init__(self, name, alpha):
        self.name = name
        self.alpha = alpha
        self.scores = []
        
    def get_q_hat(self): 
        np_arr = np.array(self.scores)
        #qhat taking the 10% quantile here
        q_hat = np.quantile(np_arr, self.alpha, method="lower")
        return q_hat


# # got this logic from the below link
# # https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):    
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def predict(model, test_image_tensor, print_bool):
    
    prediction_set = []
      
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(10, dim=1)
#         print(topk, topclass)
        # cls = class_names.class_name_lookup([topclass[0][0]]) #assigned but never used
        score = topk.cpu().numpy()[0][0]

        for i in range(10): #10 is the hardcoded num of classes
            topclass.cpu().numpy()[0][i]
            
            if print_bool: 
                print("Prediction", i+1, ":", naive_conformal_prediction.class_names.class_name_lookup([topclass.cpu().numpy()[0][i]]), ", Score: ", topk.cpu().numpy()[0][i]*100)
            
            class_num = naive_conformal_prediction.class_names.class_name_lookup([topclass.cpu().numpy()[0][i]])
            score = topk.cpu().numpy()[0][i]
            prediction_set.append((class_num, score))
            
    return prediction_set


def q_hat_scores(classObj_list): 
    calibration_threashold = []
    
    idx = 0
    for i in classObj_list: 
    #     print(classObj_list[idx].name)
    #     print(classObj_list[idx].get_q_hat())
        calibration_threashold.append((classObj_list[idx].name,classObj_list[idx].get_q_hat()))
        idx += 1
    
    return calibration_threashold

def calibrate(model, data_dir, alpha): 
    
    # calibration set has 890 images in the 10 available classes
    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std= [0.229, 0.224, 0.225])
            ])
    
    # data_dir = r'data\validation_dataset'
    dataset = ImageFolderWithPaths(root=data_dir, transform=transform)
    dataset = DataLoader(dataset=dataset)
    
    classObj_list = []
    
    model.eval() #eval mode
    
    count = 0
    
    with torch.no_grad():
        for batch in dataset:       
            
            path = batch[2]
            path = path[0] #pull path from tuple
            label_correct = path.split('\\')[2] #this is getting the label from the path 
    #         print(label_correct)         
            
            if torch.cuda.is_available():
                test_image_tensor = batch[0].view(1, 3, 224, 224).cuda()
            else:
                test_image_tensor = batch[0].view(1, 3, 224, 224)
            
            #getting the inference...softmax scores in the list
            sft_mx_scrs_list = predict(model, test_image_tensor, print_bool=False)
            
            #loop through softmax scores ensure found correct score
            name = ''
            ground_truth_score = ''
            for index, tuple in enumerate(sft_mx_scrs_list):            
                #if prediction is the correct label --> record score in object
                if sft_mx_scrs_list[index][0] == label_correct:
                    name = label_correct
                    #now get score
                    ground_truth_score = sft_mx_scrs_list[index][1]
                    
            #we now have the truth class score and softmax score        
            #if obj in list doesnt exist, make entry in the conformal prediction class list 
            idx = 0
            found_bool = False
            
            if not classObj_list == []: 
                for i in classObj_list:#if the class exists, append the softmax score to the scores list member
                    if classObj_list[idx].name == name: 
                        found_bool = True
                        classObj_list[idx].scores.append(ground_truth_score)
                    idx+=1
                    
            #if there is not already a class object matching the ground truth, make one and add the scores         
            if found_bool == False:       
                e_sub_i = Cp_Class(name, alpha)
                e_sub_i.scores.append(ground_truth_score)
                classObj_list.append(e_sub_i)
            
            count += 1
            
    # print(count)
    
    qhat_scores_list = q_hat_scores(classObj_list)
    
    return qhat_scores_list  
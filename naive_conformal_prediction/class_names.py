# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:43:42 2023

@author: WilliamKiger
"""

def class_name_lookup(key): 
    
    class_names_dict = {0: 'allergic_contact_dermatitis', 1: 'basal_cell_carcinoma', 2: 'folliculitis', 
                3: 'lichen_planus', 4: 'lupus_erythematosus', 5: 'neutrophilic_dermatoses', 
                6: 'photodermatoses', 7: 'psoriasis', 8: 'sarcoidosis', 9: 'squamous_cell_carcinoma'}
    
    key = key[0].item()
    class_name = class_names_dict.get(key)
    
    return class_name
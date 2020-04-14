# -*- coding: utf-8 -*-

from __future__ import division, print_function

import argparse
import os
import glob
import pickle
import json
import pandas as pd
import numpy as np
import scipy.io
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset

## Run the code with:
# python evaluate_template_based.py --fuse median-median --test_list_g gallery_features.json --test_list_p probe_features.json 
# python evaluate_template_based.py --fuse mean-median
# python evaluate_template_based.py --fuse none-median


def fusion(all_features_per_template, fusion_method='median'):
    all_features_per_template= np.asarray(all_features_per_template)
    if fusion_method == 'median':
        return np.median(all_features_per_template, 0)
    elif fusion_method == 'mean':
        return np.mean(all_features_per_template, 0)
    else:
        # print("No correct fusion method (i.e., mean/median) found.")
        return all_features_per_template

def Rfiw2020TestSet(x, fusion_method='median-median'):
    fuse_gallery, fuse_probe = fusion_method.split("-")
    labels = [] 
    features = [] 
    assert x in ['gallery', 'query'] 
    if x == 'gallery':
        fuse = fuse_gallery
        feat_list = opt.test_list_g
    else:
        fuse = fuse_probe 
        feat_list = opt.test_list_p       

    with open(feat_list) as file:
        data = json.load(file)
        for family_member_ind, all_features_per_template in data.items():
            label = int(family_member_ind.split('/')[0].split('F')[1])
            if fuse != "none":
                feats = fusion(all_features_per_template, fuse)
                features.append(feats)
                labels.append(label) 
            else:
                feats = all_features_per_template
                for i in range(len(all_features_per_template)):
                    features.append(all_features_per_template[i])
                    labels.append(label)    

    return np.asarray(features), np.asarray(labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1')
    parser.add_argument('--test_list_p', default='/media/yuyin/10THD1/Kinship/fiw-mm/data/lists/test/probe_features.json', type=str, help='test list probe')
    parser.add_argument('--test_list_g', default='/media/yuyin/10THD1/Kinship/fiw-mm/data/lists/test/gallery_features.json', type=str, help='test list gallery')
    parser.add_argument('--save_name', default='Rank-k_mAP', type=str, help='file name for saveing results')
    parser.add_argument('--fuse', default='median-median', type=str, help='fuse method (median/mean/none) for gallery-query')

    opt = parser.parse_args()

    str_ids = opt.gpu_ids.split(',')

    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()


    ######################################################################
    # Testing
    # ---------
    print('-------test-----------')
    
    ## Load features
    gallery_feature, gallery_label = Rfiw2020TestSet('gallery', opt.fuse)
    print("gallery size:", gallery_feature.shape, gallery_label.shape)
    
    query_feature, query_label = Rfiw2020TestSet('query', opt.fuse)
    print("query size:", query_feature.shape, query_label.shape) 

    ## Save result
    print('-->Save features to gallery_probe_features.npy')
    result = {'gallery_f': gallery_feature, 'gallery_label': gallery_label,
              'query_f'  : query_feature, 'query_label': query_label}
    np.save("gallery_probe_features.npy", result)
    

    ## Run evaluation_gpu.py
    result = './%s_result.txt' % opt.save_name
    os.system('python utils.py | tee -a %s' % result)

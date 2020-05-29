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

######################################################################
# Options
# --------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1')
    parser.add_argument('--test_feature_dir', default='/media/yuyin/10THD1/Kinship/fiw-mm/data/FIDs-MM-features', type=str, help='features of test data')
    parser.add_argument('--test_list_p', default='/media/yuyin/10THD1/Kinship/fiw-mm/data/lists/test/probes.json', type=str, help='test list probe')
    parser.add_argument('--test_list_g', default='/media/yuyin/10THD1/Kinship/fiw-mm/data/lists/test/gallery.json', type=str, help='test list gallery')
    parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
    parser.add_argument('--save_name', default='Rank-k_mAP', type=str, help='file name for saveing results')

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
    # Data
    # ---------
    class Rfiw2020TestSet(Dataset):
        def __init__(self, x):
            if x == 'gallery':
                with open(opt.test_list_g) as file:
                    self.imgs = json.load(file)
            else:
                self.imgs = []
                with open(opt.test_list_p) as file:
                    probes = json.load(file)
                    for _, family_member_ind in probes.items():
                        self.imgs.append(family_member_ind)

        def __len__(self):
            return len(self.imgs)


    ######################################################################
    # Load feature
    # ---------
    def get_gallery_feature_and_id(img_path):
        feat_path = "/media/yuyin/10THD1/Kinship/fiw-mm/data/lists/test/gallery_features.npy"        
        feat_matrix = np.loadtxt(feat_path)
        
        assert feat_matrix.shape[0] == len(img_path)
        labels = np.zeros((feat_matrix.shape[0], 1))  # size (21951, 1)
        for i, path in enumerate(img_path):
            labels[i] = int(path.split('/')[0].split('F')[1])

        return feat_matrix, labels

    def get_probe_feature_and_id(img_path):
        # size of probe img_path: 190
        labels = []
        features = []
        for path in img_path:
            label = int(path.split('/')[0].split('F')[1])
            feat_path_per_probe = os.path.join(opt.test_feature_dir, path, "encodings.pkl")
            with open(feat_path_per_probe, 'rb') as f:
                feat = pickle.load(f)
                for _, feats_per_probe in feat.items() :
                    features.append(feats_per_probe)
                    labels.append(label)

        return np.asarray(features), np.asarray(labels).reshape(-1,1)

    ######################################################################
    # Testing
    # ---------
    # Load data
    image_datasets = {x: Rfiw2020TestSet(x) for x in ['gallery', 'query']}

    print('-------test-----------')
    # Load features
    gallery_feature, gallery_label = get_gallery_feature_and_id(
        image_datasets['gallery'].imgs)
    print("gallery size:", gallery_feature.shape, gallery_label.shape)
    query_feature, query_label = get_probe_feature_and_id(
        image_datasets['query'].imgs) 
    # (4540, 512) (4540,)
    print("query size:", query_feature.shape, query_label.shape) 

    # Save result
    print('-->Save features to gallery_probe_features.npy')
    result = {'gallery_f': gallery_feature, 'gallery_label': gallery_label,
              'query_f'  : query_feature, 'query_label': query_label}
    
    np.save("gallery_probe_features.npy", result)
    

    # Run evaluation_gpu.py
    result = './%s_result.txt' % opt.save_name
    os.system('python utils.py | tee -a %s' % result)

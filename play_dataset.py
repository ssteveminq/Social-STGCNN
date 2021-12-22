#!/usr/bin/env python3
"""Load the given model and run through an environment."""

import pathlib
import yaml
from visualization import viz
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
from utils import * 
from metrics import * 
from model import social_stgcnn
import copy


paths = ['./checkpoint/*social-stgcnn*']

# for feta in range(len(paths)):
for feta in range(1):
    # ade_ls = [] 
    # fde_ls = [] 
    path = paths[feta]
    exps = glob.glob(path)
    print('Model being tested are:',exps)

    for exp_path in exps:
        print("*"*50)
        # print("Evaluating model:",exp_path)
        model_path = exp_path+'/val_best.pth'
        args_path = exp_path+'/args.pkl'
        with open(args_path,'rb') as f: 
            args = pickle.load(f)
        print("args", args)


        # stats= exp_path+'/constant_metrics.pkl'
        # with open(stats,'rb') as f: 
            # cm = pickle.load(f)
        # print("Stats:",cm)

        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        data_set = './datasets/'+args.dataset+'/'
        #original dset

        #mk
        pedset_test = PedsTrajectoryDataset(
                data_set+'test/',
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1,norm_lap_matr=True)
        #required data for plotting--MK
        peds_traj = pedset_test.peds_traj
        peds_start_ends = pedset_test.peds_start_ends
        peds_frames = pedset_test.peds_frames
        time_frames = pedset_test.time_frames

        #frame check
        save_path = pathlib.Path("gifs/"+args.dataset+".gif")
        viz.plot_ped_histories(time_frames, peds_frames, peds_traj, peds_start_ends, save_path)



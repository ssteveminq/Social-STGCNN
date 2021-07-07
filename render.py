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
import torch.distributions.multivariate_normal as torchdist
from utils import * 
from metrics import * 
from model import social_stgcnn
import copy


def render(KSTEPS=20):
    global loader_test,model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step =0 
    # print("loader_test", loader_test)
    # input("--input--")
    # peds_traj = loader_test.
    for batch in loader_test: 
        # print("batch", batch)
        # input("--batch")
        step+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch
        print("obs_taj", obs_traj)
        input("-")

        # print("obs_traj", obs_traj)
        num_of_objs = obs_traj_rel.shape[1]
        print("num_of_objs", num_of_objs)
        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp =V_obs.permute(0,3,1,2)
        # print("V_obs_tmp",V_obs_tmp )
        # input("--obs traj--")

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())
        print(V_pred.shape)
        # print("V_tr_shape", V_tr.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])
        V_pred = V_pred.permute(0,2,3,1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()


        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        print("num_of_objs", num_of_objs)
        # print("V_pred-before", V_pred)
        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]
        # print("V_pred-after", V_pred)
        print(V_pred.shape)
        print(V_tr.shape)
        # trj_only = V_tr[0,:,:]
        # print("trj_only", trj_only)
        # input("checking V")
        

        #For now I have my bi-variate parameters 
        #normx =  V_pred[:,:,0:1]
        #normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        # print("sx", sx)
        # print("sy", sy)
        corr = torch.tanh(V_pred[:,:,4]) #corr
        
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda()
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]
        # print("mean", mean)
        
        mvnormal = torchdist.MultivariateNormal(mean,cov)


        ### Rel to abs 
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 
        
        #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        # print("V_x", V_x)
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                 V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        # print("V_y", V_y)
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                 V_x[-1,:,:].copy())
        
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        # print("raw_data_dict[step]", raw_data_dict[step]['trgt'])
        # input("endfor--")

        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(KSTEPS):
            V_pred = mvnormal.sample()
            print("V_pred", V_pred)
            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
            print("V_pred_rel_to_abs", V_pred)
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            
           # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = [] 
                target = []
                obsrvs = [] 
                number_of = []
                pred.append(V_pred_rel_to_abs[:,n:n+1,:])
                # print("pred", pred)
                target.append(V_y_rel_to_abs[:,n:n+1,:])
                print("target", target)
                obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                number_of.append(1)

                ade_ls[n].append(ade(pred,target,number_of))
                fde_ls[n].append(fde(pred,target,number_of))
        
        # for n in range(num_of_objs):
            # ade_bigls.append(min(ade_ls[n]))
            # fde_bigls.append(min(fde_ls[n]))

    # ade_ = sum(ade_bigls)/len(ade_bigls)
    # fde_ = sum(fde_bigls)/len(fde_bigls)
    return True


paths = ['./checkpoint/*social-stgcnn*']
KSTEPS=20

for feta in range(len(paths)):
# for feta in range(1):
    # ade_ls = [] 
    # fde_ls = [] 
    path = paths[feta]
    exps = glob.glob(path)
    print('Model being tested are:',exps)

    for exp_path in exps:
        print("*"*50)
        print("Evaluating model:",exp_path)
        # input("--")

        model_path = exp_path+'/val_best.pth'
        args_path = exp_path+'/args.pkl'
        with open(args_path,'rb') as f: 
            args = pickle.load(f)

        stats= exp_path+'/constant_metrics.pkl'
        with open(stats,'rb') as f: 
            cm = pickle.load(f)
        print("Stats:",cm)

        obs_seq_len = args.obs_seq_len
        obs_seq_len = 1
        pred_seq_len = args.pred_seq_len
        # print("obs_seq_len", obs_seq_len)
        # print("pred_seq_len", pred_seq_len )
        data_set = './datasets/'+args.dataset+'/'
        # print("data_set", args.dataset)

        # dset_test = TrajectoryDataset(
                # data_set+'test/',
                # obs_len=obs_seq_len,
                # pred_len=pred_seq_len,
                # skip=1,norm_lap_matr=True)
        pedset_test = PedsTrajectoryDataset(
                data_set+'test/',
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1,norm_lap_matr=True)
        peds_traj = pedset_test.peds_traj
        # print("peds_traj", peds_traj)
        #required data for plotting--MK
        peds_start_ends = pedset_test.peds_start_ends
        peds_frames = pedset_test.peds_frames
        time_frames = pedset_test.time_frames

        #Defining the model 
        # model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
        # output_feat=args.output_size,seq_len=args.obs_seq_len,
        # kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()
        # model.load_state_dict(torch.load(model_path))

        # print("Testing ....")
        # render()


        save_path = pathlib.Path("gifs/"+args.dataset+".gif")
        viz.plot_ped_histories(time_frames, peds_frames, peds_traj , peds_start_ends, save_path)



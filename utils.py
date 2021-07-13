import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time

ColorSet=['red', 'green', 'blue','yellow', 'black', 'cyan' ]

def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)
                
def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    seq_ = seq_.squeeze()
    # print("seq_",seq_ )
    # input("seq")
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr: 
            G = nx.from_numpy_matrix(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data) 

class PedsTrajectoryDataset(): 
    def __init__( self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002, min_ped=1, delim='\t',norm_lap_matr = True):
        """ Args: - data_dir: Directory containing dataset files in the format <frame_id> <ped_id> <x> <y> - obs_len: Number of time-steps in input trajectories 
        - pred_len: Number of time-steps in output trajectories 
        - skip: Number of frames to skip while making the dataset - threshold: Minimum error to be considered for non linear traj when using a linear predictor 
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files 
        """ # super(TrajectoryDataset, self).__init__() 
        self.max_peds_in_frame = 0 
        self.data_dir = data_dir 
        self.obs_len = obs_len 
        self.pred_len = pred_len
        self.skip = skip 
        self.seq_len = self.obs_len + self.pred_len 
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files] 
        self.peds_traj=[]
        self.peds_start_ends=[] 
        self.peds_frames=[] 
        self.time_frames=[]
        for path in all_files:
            data = read_file(path, delim) 
            # print("data", data)
            # input("data")
            frames = np.unique(data[:, 0]).tolist() 
            self.time_frames=frames
            numpeds= np.unique(data[:, 1]).tolist() 
            # print("numpeds", numpeds)
            max_num_peds = int(max(numpeds)) 
            frame_len=int(max(frames))
            # print("numpeds", max_num_peds) 
            #sort by time frames
            frame_data = [] 
            peds_data=dict() #mk 
            peds_start_ends=dict() 
            peds_frames=[]
            #for i in range(max_num_peds): 
            # peds_data[i+1]=[0,  0]
            # print("peds_data", peds_data) 
            for frame in frames: 
                frame_data.append(data[frame == data[:, 0], :])
                # print("frame_data", frame_data)
                # input("--")
                # for idx in range(0, num_sequences * self.skip + 1, skip):
            for idx, time_frame in enumerate(frames):
                if idx <= len(frame_data):
                    curr_seq_data = frame_data[idx] 
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) 
                    # peds_frames.append(peds_in_curr_seq )
                    # self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq)) 
                    # curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    # print("peds_in_curr_seq", peds_in_curr_seq) 
                    for _, ped_id in enumerate(peds_in_curr_seq): # print("ped_id", ped_id)
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                        curr_ped_seq = curr_ped_seq[:,2:4][0]
                        if ped_id in peds_data.keys():
                            peds_data[ped_id]= np.vstack([peds_data[ped_id], curr_ped_seq]) 
                            peds_start_ends[ped_id][1]=time_frame# print("peds_start_ends", peds_start_ends) # input("--") 
                            # print("time_frame-end_update", time_frame)
                            
                        else: 
                            peds_data[ped_id]=curr_ped_seq
                            peds_start_ends[ped_id]=[time_frame,time_frame]
                            # print("time_frame", time_frame)

                    #peds_data format key: [ped_id] , value: trajectories
                    self.peds_traj=peds_data
                    # print("self.peds_traj", self.peds_traj)
                    #peds_start_ends format key: [ped_id] , value: [start_time_idx, end_time_idx]
                    self.peds_start_ends=peds_start_ends
                    self.peds_frames.append(peds_in_curr_seq)



class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))
            # print("self.seq_len", self.seq_len)--20

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                # print("peds_in_curr_seq", peds_in_curr_seq )
                #peds_in_curr_seq = > current number of pedestirans for each sequence
                #number of peds
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                # print("len--peds_in_curr_seq", len(peds_in_curr_seq))
                # input("--")

                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                #iteration for individual pedestrian
                for _, ped_id in enumerate(peds_in_curr_seq):
                    #gather data only for specific ped_id
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    #what is curr_ped_seq[0,0]
                    #what is curr_ped_seq[-1,0]
                    # print("curr_ped_seq", curr_ped_seq)
                    # print("curr_ped_seq[0,0]", curr_ped_seq[0,0])
                    # print("curr_ped_seq[-1,0]", curr_ped_seq[-1,0])
                    # input("--11--")
                    # the begining time index when firstly ped comes
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    # the lasttime index when the observation of ped 
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    # print("pad_front", pad_front)
                    # print("pad_end", pad_end)
                    #do not train when the length of sequence of pedestrian less than seq_len (20)
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    # print("curr_ped_seq", curr_ped_seq)
                    #collect x-y coordinates
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative--why??
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    # print("rel_currr_ped_seq", rel_curr_ped_seq)
                    _idx = num_peds_considered
                    # print("_idx", _idx)
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # print("curr_ped_seq_rel ", curr_ped_seq_rel )
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    # print("curr_seq", curr_seq[:num_peds_considered])
                    # print("curr_seq_rel", curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        # print("seq_list", seq_list)
        # input("--")
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        # print("cum_start_idx", cum_start_idx)
        # input("--")
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        print("--graph before--")
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.v_pred = [] 
        self.A_pred = [] 
        # print("self.seq_start_end", self.seq_start_end)
        # print("len(self.seq_start_end)", len(self.seq_start_end))
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]
            # print("start", start)
            # print("end", end)
            # input("--")

            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        return out



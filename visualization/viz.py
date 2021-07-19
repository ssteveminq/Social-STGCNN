"""A library of functions for taking in a game history and plotting
the results to a gif."""

import collections
import pathlib

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

ColorSet=['red', 'green', 'blue','yellow', 'black', 'cyan', 'magenta' ]


def plot_state(idx, robot_state: torch.Tensor, human_states: torch.Tensor):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        [robot_state[0, 0].item()],
        [robot_state[0, 1].item()],
        marker="o",
        markersize=3,
        color="red",
    )
    sesnor_circle= plt.Circle((robot_state[0, 0].item(), robot_state[0, 1].item()), 5.0, color='g', fill=False)
    ax.add_patch(sesnor_circle)
    ax.plot(
        [robot_state[0, 2].item()],
        [robot_state[0, 3].item()],
        marker="o",
        markersize=3,
        color="green",
    )
    for human in human_states:
        ax.plot(
            [human[0].item()],
            [human[1].item()],
            marker="o",
            markersize=3,
            color="blue",
        )
    ax.set_ylim(-25, 25)
    ax.set_xlim(-25, 25)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def plot_trajectories(idx, peds_in_frame, peds_traj, preds_trj, peds_start_ends ):
    fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(
        # [robot_state[0, 0].item()],
        # [robot_state[0, 1].item()],
        # marker="o",
        # markersize=3,
        # color="red",
    # )
    # sesnor_circle= plt.Circle((robot_state[0, 0].item(), robot_state[0, 1].item()), 5.0, color='g', fill=False)
    # ax.add_patch(sesnor_circle)
    # ax.plot(
        # [robot_state[0, 2].item()],
        # [robot_state[0, 3].item()],
        # marker="o",
        # markersize=3,
        # color="green",
    # )
    # print("peds_traj", peds_traj)
    # print("idx", idx)
    # input("--")
    for ped_idx, ped in enumerate(peds_in_frame):
        if ped in peds_traj.keys():
            pos = peds_traj.get(ped)
            start = peds_start_ends.get(ped)[0]
            end= peds_start_ends.get(ped)[1]
            cur_idx = int((idx-start)/10.0)
            # print("cur_idx", cur_idx)
            # print("ped", ped)
            if cur_idx < len(pos):
                # print("cur_idx", cur_idx)
                # print("pos", pos)
                if cur_idx>0:
                    # print("pos--", pos)
                    # print("pos[0]--", pos[0])
                    # print("pos[1]--", pos[1])
                    # print("pos[0:2-x]--", pos[0:3,0])
                    # print("pos[0:2-y]--", pos[0:3,1])
                    pos=pos[0:int(cur_idx+1)]
                    pos_x = pos[0:int(cur_idx+1),0]
                    pos_y = pos[0:int(cur_idx+1),1]
                    # print("cur_idx", cur_idx)
                    # print("--------pos--", pos)
                    # input("--")
                elif cur_idx==0:
                    pos=pos[int(cur_idx)]
                    pos_x = pos[0]
                    pos_y = pos[1]
                else:
                    continue
            else:
                # print("cur_idx: ", cur_idx, "end_idx", end)
                continue
        #check time index in peds_tart_ends
        # pos= peds_traj[int(ped)]
        # print("pos", pos)
        # input("---")
        color_idx = int(ped%7)
        if ped_idx<len(preds_trj):
            pred_xs = preds_trj[ped_idx]
            if len(pred_xs)==1:
            # print("len_pred_xs[0]", len(pred_xs[0]))
            # print("len_pred_xs[0][:,0]", len(pred_xs[0][:,0]))
                pred_xcoords = pred_xs[0][:,0][:,0]
                pred_ycoords = pred_xs[0][:,0][:,1]
                ax.plot(pred_xcoords,pred_ycoords,
                marker="*",
                markersize=7,
                color=ColorSet[color_idx],
                alpha=0.5
                )

        # print("pred_xs", pred_xs)
        # print("pred_xs-0", pred_xs[0][0][0])
        # print("pred_xs-2", pred_xs[0][:,0][:,0])
        # print("pred_xs-2", pred_xs[0][:,0][:,0])
        # input("--")
        # print("pred_xs", pred_xs[0:])
        # pred_xs = preds_trj[ped_idx]
        
        # print("pos[0]", pos[0])
        # print("pos[1]", pos[1])
        ax.plot(pos_x,pos_y,
            marker="o",
            markersize=10,
            color=ColorSet[color_idx],
        )
        
    ax.set_ylim(-2, 16)
    ax.set_xlim(-2, 16)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image




def plot_history(game_history, save_path: pathlib.Path):
    save_path.parent.mkdir(exist_ok=True, parents=True)
    robot_states = [state["robot_states"] for state in game_history]
    human_states = [state["human_states"] for state in game_history]

    imageio.mimsave(
        str(save_path),
        [
            plot_state(idx, robot_state, human_states)
            for idx, (robot_state, human_states) in enumerate(
                zip(robot_states, human_states)
            )
        ],
        fps=15,
    )

#for each frame, get peds in that sequence, plot the peds
def plot_ped_histories(time_frames, peds_frames, peds_traj ,predicted_traj, peds_start_ends, save_path: pathlib.Path):
    save_path.parent.mkdir(exist_ok=True, parents=True)
    #get pedestrians in time frame
    #current time frame: 
    offset= len(time_frames)-len(predicted_traj)
    for k in range(offset):
        offset_tmp={}
        for j in range(2):
            offset_tmp[j]=[]
            offset_tmp[j].append(np.zeros((2,1)))
            offset_tmp[j].append(np.zeros((2,1)))
        predicted_traj.append(offset_tmp)

    print("len(time_frames)", len(time_frames))
    print("len(mk_preds)", len(predicted_traj))
    # input("--")
    imageio.mimsave(
        str(save_path),
        [
           plot_trajectories(time_frame, peds_frames[time_idx], peds_traj, predicted_traj[time_idx],peds_start_ends)
            for time_idx, time_frame in enumerate(time_frames)
        ],
        fps=2,
    )

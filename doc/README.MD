document

#### Class PedsTrajetoryDataset
- Input: Dataset (test/)
- Output:peds_traj, peds_start_ends, peds_frames, time_frames

#### Data management <br/>
1.peds_traj
  - Dictonary format: (key: [ped_id] , value: trajectories)
  - trajectories: [ (x1, y1), (x2, y2), ... (xn, yn) ]

2.peds_start_ends
  - Dictonary format: (key: [ped_id] , value: [start time, end_time])

3. peds_frames
  - [time_stamp, peds_list]

4. time_frames
  - list of time_stamp



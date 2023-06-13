# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Andreas Look, Andreas.Look@de.bosch.com

import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser('Preprocess rounD')
parser.add_argument('--path_to_raw_data')
parser.add_argument('--path_to_processed_data')
args = parser.parse_args()

if __name__ == "__main__":    
    path_to_raw_data = args.path_to_raw_data
    path_to_processed_data = args.path_to_processed_data
    
    f = 25 # frequency
    overlap = 100 # overlap of each snippet
    len_snippet = 200 # we construct chunks with 200 time steps = 8 seconds
    thresshold=30 # connect all vehicles which are closer than thresshold meters
    eligable_classes = ["car", "van", "truck", "bus", "motorcycle"]
    
    if not os.path.isdir(path_to_processed_data):
        os.mkdir(path_to_processed_data)
        
    for recordingId in range(24):
        recordingId = str(recordingId).zfill(2)
        print("Construction data matrix of scenario: {}".format(recordingId))

        path_to_tracksmeta = os.path.join(path_to_raw_data, "{}_tracksMeta.csv".format(str(recordingId).zfill(2)))
        path_to_tracks = os.path.join(path_to_raw_data, "{}_tracks.csv".format(str(recordingId).zfill(2)))

        csv = pd.read_csv(path_to_tracksmeta)
        track = pd.read_csv(path_to_tracks)

        class_mask = []
        driving_mask = []
        for value in csv["class"].values: # remove vehicles that are not in eligable_classes
            if value in eligable_classes:
                class_mask.append(True)
            else:
                class_mask.append(False)
        for value in csv["numFrames"].values: # remove vehicles that are not moving
            if value < 1000:
                driving_mask.append(True)
            else:
                driving_mask.append(False)

        mask = np.array(class_mask) & np.array(driving_mask)
        csv_filtered = csv[mask]
        numframes = max(csv_filtered["finalFrame"].values)
        num_chunks = int(numframes/overlap)

        scenario = []
        for chunk in range(num_chunks):
            startframe = chunk*overlap
            csv_mask_initial = csv_filtered["initialFrame"].values <= startframe
            csv_mask_final = csv_filtered["finalFrame"].values >= startframe
            csv_mask = csv_mask_final & csv_mask_initial

            csv_chunk = csv_filtered[csv_mask]
            eligable_Ids = csv_chunk["trackId"].values
            traj = []
            for Id in eligable_Ids:
                track_mask_Id = track["trackId"]==Id
                track_Id = track[track_mask_Id]
                traj_Id = track_Id[["xCenter", "yCenter"]].values
                start_arg = np.argwhere(track_Id["frame"].values==startframe)[0,0]
                end_arg = start_arg + int(len_snippet)
                traj_Id_chunk = traj_Id[start_arg:end_arg]
                pad_len = int(len_snippet)-len(traj_Id_chunk)
                traj_Id_chunk_pad = np.pad(traj_Id_chunk, ((0,pad_len),(0,0)), mode="constant")
                traj.append(traj_Id_chunk_pad)
            traj = np.array(traj, dtype=np.float32)
            if len(traj.shape) == 3:  # append traj if it is not empty
                pad_len = 22-traj.shape[0] # 22 is the number of maximum agents
                traj_pad = np.pad(traj, ((0,pad_len),(0,0),(0,0)), mode="constant")
                scenario.append(traj_pad)
        scenario = np.array(scenario)
        path_to_processed_tracks = os.path.join(path_to_processed_data, "scenario_{}.npy".format(recordingId))
        np.save(path_to_processed_tracks, scenario)
        
    for recordingId in range(24):
        recordingId = str(recordingId).zfill(2)
        print("Construction adjencency matrix of scenario: {}".format(recordingId))
        path_to_processed_tracks = os.path.join(path_to_processed_data, "scenario_{}.npy".format(recordingId))
        scenario = np.load(path_to_processed_tracks,allow_pickle=False)

        A = []
        for chunk in scenario:
            a = np.zeros((22,22))
            chunk_initial_step = chunk[:,0]
            num_valid_vehicles = ((chunk_initial_step==0).sum(1)==0).sum()
            sub_chunk_initial_step = chunk_initial_step[:num_valid_vehicles]
            delta_matrix = np.expand_dims(sub_chunk_initial_step, 1) - np.expand_dims(sub_chunk_initial_step, 0)
            l2_matrix = np.sqrt(np.mean(delta_matrix**2, 2))
            a[:num_valid_vehicles, : num_valid_vehicles] = (l2_matrix<thresshold).astype(np.int)
            A.append(a)
        A = np.array(A)
        path_to_adjacency = os.path.join(path_to_processed_data, "adjacency_{}.npy".format(recordingId))
        np.save(path_to_adjacency, A)
        
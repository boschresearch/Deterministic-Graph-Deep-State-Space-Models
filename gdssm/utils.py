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
import torch

import numpy as np

def wrap_mP(m, P, mode):
    """Extract for each agent a prediction.
        
    Args:
        m: Mean with shape (horizon, batch_size, num_nodes, D).
        P: Covariance with shape 
            if `mode` is `full`: 
                (horizon, batch_size, num_nodes , D, num_nodes, D),
            if `mode` is `major_diagonal`: 
                (horizon, batch_size, num_nodes, D),
            if `mode` is `major_blocks`:
                (horizon, batch_size, num_nodes, D, D),
            if `mode` is `all_diagonal`:
                (horizon, batch_size, num_nodes, D, num_nodes).
    
    Returns:
        m_out: Mean with shape (batch_size, horizon, num_nodes, D)
        P_out: Covariance with shape (batch_size, horizon, num_nodes, D, D)
    """
    m_out = m.transpose(0,1)
    if mode=="full":
        P_out = torch.diagonal(P.transpose(0,1), 0, 2,4).transpose(2,4) 
    if mode=="major_diagonal":
        P_out = torch.diag_embed(P.transpose(0,1))
    if mode=="major_block":
        P_out = P.transpose(0,1)
    if mode=="all_diagonal":
        P_out = torch.diagonal(P.transpose(0,1),0,2,4).transpose(2,3)
        P_out = torch.diag_embed(P_out)
    return m_out, P_out

def NLL(observation, m, P, jitter=1e-1):
    """Returns Negative Log-Likelihood.
    
    Args:
        observation: Observation with shape (batch_size, D)   
        m: Mean with shape (batch_size, D).
        P: Covariance with shape (batch_size, D, D).
        jitter: Is added to the main diagonal of `P` for numerical stability.
        
     Returns:
        nll: Negative Log-Likelihood for each observation with shape (batch_size).
    """
    dim = m.shape[1]
    device = observation.device
    const = torch.FloatTensor([np.log(2*np.pi)*dim/2]).to(device)
    
    batch_size, observation_space = observation.shape
    I = torch.eye(observation_space).unsqueeze(0).to(device)
    
    chol = torch.linalg.cholesky(P+I*jitter) 

    B = I.expand(batch_size, observation_space, observation_space)
    inv = torch.linalg.solve(chol, B) # inverse via solving system of equations
    inv = inv.transpose(1,2)@inv  

    det = torch.prod(torch.diagonal(chol, 0, 1, 2)**2, 1).view(batch_size, 1, 1) 

    l2_distance = (observation - m).view(batch_size, 1, observation_space)
    mhb = l2_distance@inv@l2_distance.transpose(1,2) # mahabolis distance

    
    nll =  0.5*(mhb + torch.log(det)) + const
    return nll

class RounDDataset(torch.utils.data.Dataset):
    """A PyTorch dataset class for RounD.

    Args:
        path_A: Path to adjancency matrices.
        path_D: Path to tracks.
        scenario_Ids: A list that contains all scenarios to use.
        down_sampling: Factor by which data is downsampled.
    """
    def __init__(self, path_A, path_D, scenario_Ids, down_sampling=5):
        self.path_A = path_A
        self.path_D = path_D
        self.scenario_Ids = scenario_Ids
        self.down_sampling = down_sampling
    
        self.time_hist = 3# length of hist traj in s 
        self.f = 25 # sampling rate
    
        self._load_A()
        self._load_D()
    
    def _load_A(self):
        A = []
        for scenario_Id in self.scenario_Ids:
            scenario_Id = str(scenario_Id).zfill(2)
            path_to_A_ = os.path.join(self.path_A, "adjacency_{}.npy".format(scenario_Id))
            A_ = np.load(path_to_A_)
            A.extend(A_)
        self.A = torch.from_numpy(np.array(A).astype(np.float32))
            
    def _load_D(self):
        D = []
        for scenario_Id in self.scenario_Ids:
            scenario_Id = str(scenario_Id).zfill(2)
            path_to_D_ = os.path.join(self.path_D, "scenario_{}.npy".format(scenario_Id))
            D_ = np.load(path_to_D_)
            D.extend(D_)
        self.D = torch.from_numpy(np.array(D))
    
    def __len__(self):
        return len(self.D)
    
    def __getitem__(self, idx):
        traj = self.D[idx]
        hist = traj[:,:self.f*self.time_hist]
        hist = hist[:,::self.down_sampling]
        
        fut = traj[:,self.f*self.time_hist:]
        fut = fut[:,::self.down_sampling]
        
        A = self.A[idx]
        return hist, fut, A 


class ToyDataset(torch.utils.data.Dataset):
    """ Toy dataset.
    
    Args:
        X: dataset with shape (num_rollouts, timesteps, dim).
    """
    
    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, idx):
        return self.X[idx]

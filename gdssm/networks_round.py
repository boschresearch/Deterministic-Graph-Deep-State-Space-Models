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

import torch
from .ssm import BaseGDSSM
from .layers import Linear, ReLU, MeanAggregator


class Dynamics(BaseGDSSM):
    def __init__(self, dt, D_x, mode="full", D_hidden=24, mask=None):
        super().__init__(mode=mode, dt=dt)
        
        self.D_x = D_x
        self.D_hidden = D_hidden
        
        self.f1 = Linear(D_in=D_x*2, D_out=D_hidden, mode=mode)
        self.f2 = Linear(D_in=D_hidden, D_out=D_hidden, mode=mode)
        self.f3 = Linear(D_in=D_hidden, D_out=D_x, mode=mode)
        
        self.L1 = Linear(D_in=D_x*2, D_out=D_hidden, mode=mode)
        self.L2 = Linear(D_in=D_hidden, D_out=D_x, mode=mode)
        
        self.relu = ReLU(mode=mode)
        self.aggregator = MeanAggregator(concat=True, mode=mode)
        
    def drift(self, x, A): 
        x = self.aggregator(x, A)
        x = self.f1(x, A)
        x = self.relu(x, A)
        x = self.f2(x, A)
        x = self.relu(x, A)
        x = self.f3(x, A)
        return x
    
    def drift_moments(self, m, P, A, new_batch=True):
        jac = self.aggregator.jac(m, A, new_batch)
        m, P = self.aggregator.next_moments(m, P, A, new_batch)
        
        jac = self.f1.jac(m, new_batch)@jac
        m, P = self.f1.next_moments(m, P, new_batch)
        
        jac = self.relu.jac(m, P)@jac
        m, P = self.relu.next_moments(m, P)
        
        jac = self.f2.jac(m, new_batch)@jac
        m, P = self.f2.next_moments(m, P, new_batch)
        
        jac = self.relu.jac(m, P)@jac
        m, P = self.relu.next_moments(m, P)
        
        jac = self.f3.jac(m, new_batch)@jac
        m, P = self.f3.next_moments(m, P, new_batch)
        
        self.exp_jac = jac
        return m, P
        
    def diffusion(self, x, A):
        x = self.aggregator(x, A)
        x = self.L1(x, A)
        x = self.relu(x, A)
        x = self.L2(x, A)
        x = self.relu(x, A)
        return x
    
    def diffusion_moments(self, m, P, A, new_batch):
        m, P = self.aggregator.next_moments(m, P, A, new_batch)
        m, P = self.L1.next_moments(m, P, new_batch)
        m, P = self.relu.next_moments(m, P, new_batch)
        m, P = self.L2.next_moments(m, P, new_batch)
        m, P = self.relu.next_moments(m, P, new_batch)       
        return m, P
    
class Decoder(torch.nn.Module):
    def __init__(self, D_e, D_x=2, mode="full", D_hidden=24):
        super().__init__()
        self.D_hidden = D_hidden
        self.D_e = D_e
        self.D_x = D_x
        self.mode=mode
        
        self.f1 = Linear(D_in=D_e, D_out=D_hidden, mode=mode)
        self.f2 = Linear(D_in=D_hidden, D_out=D_x, mode=mode)
        self.relu = ReLU(mode=mode)
    
        self.log_noise = torch.nn.Parameter(torch.FloatTensor([-1]))
    
    def embed_P_diag(self,P_diag):
        batch_size, num_nodes, D_x = P_diag.shape
        if self.mode =="full":
            P = torch.diag_embed(P_diag.reshape(batch_size, -1))
            P = P.reshape(batch_size, num_nodes, D_x, num_nodes, D_x)
        if self.mode=="major_block":
            P = torch.diag_embed(P_diag)
        if self.mode=="all_diagonal":
            P = torch.diag_embed(P_diag.transpose(1,2)).transpose(1,2)
        if self.mode=="major_diagonal":
            P = P_diag
        return P
        
    def next_moments(self, m, P, new_batch=True, *args):
        device = m.device
        m, P = self.f1.next_moments(m, P, new_batch)
        m, P = self.relu.next_moments(m, P, new_batch)
        m, P = self.f2.next_moments(m, P, new_batch)
        
        P_observation_diag = torch.exp(self.log_noise)*torch.ones(m.shape).to(device)
        P = P + self.embed_P_diag(P_observation_diag)
        return m, P
    
    
    def forward(self, x, *ags):
        device = x.device 
        x_dec = self.f1(x)
        x_dec = self.relu(x_dec)
        x_dec = self.f2(x_dec)
        noise = torch.sqrt(torch.exp(self.log_noise))*torch.randn(x_dec.shape).to(device)
        return x_dec + noise
    
class Encoder(torch.nn.Module):
    def __init__(self, D_e, D_x=30, num_modes=1):
        super().__init__() 
        self.D_e = D_e
        self.D_x = D_x
        self.D_h = 64
        self.num_modes = num_modes
        
        self.f_emb = torch.nn.Linear(self.D_x, self.D_h, bias=True)
        self.f_msg_1 = torch.nn.Linear(self.D_h+self.D_h, self.D_h, bias=True)
        
        self.mean_layer = torch.nn.Linear(self.D_h, self.D_e*num_modes, bias=True)
        self.cov_layer = torch.nn.Linear(self.D_h, self.D_e*num_modes, bias=True)
        self.weight_layer = torch.nn.Linear(self.D_h, num_modes, bias=True)
        
        
    def aggregate(self, v, A):
        normalizer = A.sum(2).unsqueeze(-1).clamp(1,100)

        v_agg = (A@v)/normalizer
        return v_agg
    
    def forward(self, x, A):
        batch_size, num_nodes, horizon, features = x.shape
        x = x.reshape(batch_size, num_nodes, horizon*features)
        
        x_emb = torch.tanh(self.f_emb(x.reshape(batch_size*num_nodes, self.D_x)))
        x_emb = x_emb.reshape(batch_size, num_nodes, self.D_h)
        
        x_msg = self.aggregate(x_emb, A)
        
        x_enc = torch.cat((x_msg, x_emb), 2).reshape(batch_size*num_nodes, self.D_h+self.D_h)
        x_enc = torch.tanh(self.f_msg_1(x_enc))
        
        m_enc = self.mean_layer(x_enc).reshape(batch_size, num_nodes, self.num_modes, self.D_e)
        m_enc = m_enc.transpose(1,2)
        P_enc = torch.exp(self.cov_layer(x_enc)).reshape(batch_size, num_nodes, self.num_modes, self.D_e)
        P_enc = P_enc.transpose(1,2).reshape(batch_size, self.num_modes, num_nodes*self.D_e)
        P_enc = torch.diag_embed(P_enc)
        P_enc = P_enc.reshape(batch_size, self.num_modes, num_nodes, self.D_e, num_nodes, self.D_e)
              
        weights = self.weight_layer(x_enc).reshape(batch_size, num_nodes, self.num_modes)
        weights = torch.nn.functional.softmax(weights.mean(1), 1)
        
        return m_enc, P_enc, weights
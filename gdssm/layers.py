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
#
# The source code for the class ReLU and Heaviside is derived from 
# Deterministic Variational Inferencen (https://github.com/microsoft/deterministic-variational-inference)
# Copyright (c) 2018 Microsoft Corporation. All rights reserved.
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
   
class Mask(torch.nn.Module):
    """Helper class to extract relevant values from covariance matrix.

    Args:
        mask_type: Type of covariance approximation
        num_nodes: Number of nodes in the graph
        D: Dimensionality  
    """
    def __init__(self, mask_type, D, num_nodes):
        super().__init__()
        self.mask_type = mask_type
        self.num_nodes = num_nodes
        self.D = D
        
        self._build_blocks()
                   
    def _build_blocks(self):
        """Contruct the mask for covariance matrix. Is 1 where entry is valid, 0 otherwise."""
        if self.mask_type=="full":
            major_blocks = torch.ones(self.D, self.D)
            minor_blocks = torch.ones(self.D, self.D)
        elif self.mask_type=="major_diagonal":
            major_blocks = torch.eye(self.D)
            minor_blocks = torch.zeros(self.D, self.D)
        elif self.mask_type=="all_diagonal":
            major_blocks = torch.eye(self.D)
            minor_blocks = torch.eye(self.D)
        elif self.mask_type=="major_block":
            major_blocks = torch.ones(self.D, self.D)
            minor_blocks = torch.zeros(self.D, self.D)
        elif self.mask_type=="major_block_minor_diagonal":
            major_blocks = torch.ones(self.D, self.D)
            minor_blocks = torch.eye(self.D)
        self.major_blocks = major_blocks
        self.minor_blocks = minor_blocks
        
        self._expand_major_blocks()
        self._expand_minor_blocks()
        
        self.register_buffer('value', self.minor_blocks + self.major_blocks)
    
    def _expand_minor_blocks(self):
        """Mask values on off-diagonal"""
        block_1 = self.minor_blocks.repeat(self.num_nodes, self.num_nodes).unsqueeze(0)
        
        I = torch.eye(self.num_nodes).unsqueeze(0)
        block_2 = self.minor_blocks.unsqueeze(0)
        block_2 = self._kronecker(block_2, I)
        block_2 = block_2.reshape(1, self.D*self.num_nodes, self.D*self.num_nodes)
        
        self.minor_blocks = block_1 - block_2
             
    def _expand_major_blocks(self):
        """Mask values on main diagonal"""
        I = torch.eye(self.num_nodes).unsqueeze(0)
        self.major_blocks = self.major_blocks.unsqueeze(0)
        self.major_blocks = self._kronecker(self.major_blocks, I)
        self.major_blocks = self.major_blocks.reshape(1, 
                                                      self.D*self.num_nodes, 
                                                      self.D*self.num_nodes)
             
    def _kronecker(self,A, B):
        """Kronecker product"""
        return torch.einsum("bij,bkl->blikj", A, B)
     
    def filter_P(self,P):
        """Filter torch covariance matrix.        
        
        Args:
            P: Covariance with shape (batch_size, num_nodes , D, num_nodes, D).
                
        Returns:
            P_filtered: Covariance with shape 
                if `mask_type` is `full`: 
                    (batch_size, num_nodes , D, num_nodes, D),
                if `mask_type` is `major_diagonal`: 
                    (batch_size, num_nodes, D),
                if `mask_type` is `major_blocks`:
                    (batch_size, num_nodes, D, D),
                if `mask_type` is `all_diagonal`:
                    (batch_size, num_nodes, D, num_nodes).
        """
        batch_size, num_nodes, D, _, _ = P.shape
        D_tot = num_nodes*D
            
        if self.mask_type=="major_diagonal":
            P = P.reshape(batch_size, D_tot, D_tot)
            P = torch.diagonal(P,0,1,2).reshape(batch_size, num_nodes, D)
        if self.mask_type=="all_diagonal":
            idx = self.value.bool().repeat_interleave(batch_size, 0)
            P = P.reshape(batch_size, D_tot, D_tot)
            P = P[idx].reshape(batch_size, num_nodes, D, num_nodes)
        if self.mask_type=="full":
            P = P
        if self.mask_type=="major_block":
            idx = self.value.bool().repeat_interleave(batch_size, 0)
            P = P.reshape(batch_size, D_tot, D_tot)
            P = P[idx].reshape(batch_size, num_nodes, D, D)
        return P     
    
    def JP_product(self, J, P):
        """Calculate Jacobian covariance product. 
         Args:
            J: Jacobian with shape (batch_size, num_nodes*D, num_nodes*D).
            P: Covariance with shape
                if `mask_type` is `full`: 
                    (batch_size, num_nodes , D, num_nodes, D),
                if `mask_type` is `major_diagonal`: 
                    (batch_size, num_nodes, D),
                if `mask_type` is `major_blocks`:
                    (batch_size, num_nodes, D, D),
                if `mask_type` is `all_diagonal`:
                    (batch_size, num_nodes, D, num_nodes).    
        Returns:
            JP: Product between covariance and Jacobian with shape
                if `mask_type` is `full`: 
                    (batch_size, num_nodes , D, num_nodes, D),
                if `mask_type` is `major_diagonal`: 
                    (batch_size, num_nodes, D),
                if `mask_type` is `major_blocks`:
                    (batch_size, num_nodes, D, D),
                if `mask_type` is `all_diagonal`:
                    (batch_size, num_nodes, D, num_nodes).
        """
        if self.mask_type == "all_diagonal":
            batch_size, num_nodes, D, _ = P.shape
            D_tot = num_nodes*D
            P_pad = torch.diag_embed(P.transpose(2,3)).transpose(2,3)
            JP = J@P_pad.reshape(batch_size, D_tot, D_tot)
            JP = JP[self.value.bool().repeat_interleave(batch_size, 0)].reshape(P.shape)
            return JP

        if self.mask_type =="full":
            batch_size, num_nodes, D, _, _ = P.shape
            D_tot = num_nodes*D
            P = P.reshape(batch_size, D_tot, D_tot)
            JP = J@P
            return JP.reshape(batch_size, num_nodes, D, num_nodes, D)

        if self.mask_type == "major_block":
            batch_size, num_nodes, D, _ = P.shape
            J_reshaped = J[self.value.bool().repeat_interleave(batch_size, 0)].reshape(P.shape)
            JP = (J_reshaped.reshape(-1, D, D)@P.reshape(-1, D, D)).reshape(P.shape)
            return JP

        if self.mask_type == "major_diagonal":
            batch_size, num_nodes, D = P.shape
            JP = torch.diagonal(J, 0, 1, 2)*P.reshape(batch_size, num_nodes*D)
            return JP.reshape(batch_size, num_nodes, D)
    
    
class Linear(torch.nn.Module):
    """Linear Layer.

    Args:
        D_in: Number of input features.
        D_out: Number of output features.
        bias: If `True` use bias.
        mode: Covariance approximation mode.
    """
    def __init__(self, D_in, D_out, bias=False, mode="full"):
        super().__init__()
        
        self.eligable_modes = ["full", 
                               "major_block",
                               "all_diagonal",
                               "major_diagonal"]
        assert mode in self.eligable_modes, "wrong mode"
        
        self.D_in = D_in
        self.D_out = D_out
        self.bias = bias
        self.mode = mode
        
        self.f = torch.nn.Linear(D_in, D_out, bias=bias)
        
    def _kronecker(self, A, B):
        return torch.einsum("bij,bkl->blikj", A, B)
    
    def _build_A(self, m):
        """Contruct the augmented weight matrix."""
        batch_size, num_nodes, D_in = m.shape  
        device = m.device
        
        A = self.f.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        I = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        A_expanded = self._kronecker(A,I).reshape(batch_size, A.size(1)*I.size(1), A.size(2)*I.size(2))
        return A_expanded
        
    def jac(self, m, new_batch=True, *args):
        """Return the Jacobian.
        
        Args:
            m: Mean with shape (batch_size, num_nodes, D_in).
            new_batch: If `True` recalculate the augmented weight matrix.
                
        Returns:
            J: Jacobian shape `(batch_size, num_nodes*D_out, num_nodes*D_in)`,
        """
        if new_batch:
            self._jac = self._build_A(m)
        return self._jac
        
    def next_moments(self, m, P, new_batch=True, *args):
        """Calculate the next moments.
        
        Args:
            m: Mean with shape (batch_size, num_nodes, D_in).
            P: Covariance with shape 
                if `mode` is `full`: 
                    (batch_size, num_nodes , D_in, num_nodes, D_in),
                if `mode` is `major_diagonal`: 
                    (batch_size, num_nodes, D_in),
                if `mode` is `major_blocks`:
                    (batch_size, num_nodes, D_in, D_in),
                if `mode` is `all_diagonal`:
                    (batch_size, num_nodes, D_in, num_nodes).
            new_batch: If `True` recalculate the augmented weight matrix.
                
        Returns:
            next_m: Mean with shape (batch_size, num_nodes, D_out).
            next_P: Covariance with shape 
                if `mode` is `full`: 
                    (batch_size, num_nodes , D_out, num_nodes, D_out),
                if `mode` is `major_diagonal`: 
                    (batch_size, num_nodes, D_out),
                if `mode` is `major_blocks`:
                    (batch_size, num_nodes, D_out, D_out),
                if `mode` is `all_diagonal`:
                    (batch_size, num_nodes, D_out, num_nodes).
        """
        batch_size, num_nodes, D_in = m.shape    
        next_m = self.f.weight.unsqueeze(0)@(m.reshape(batch_size, -1, D_in).transpose(1,2))
        next_m = next_m.transpose(1,2).reshape(batch_size, num_nodes, -1)
        if self.bias:
            b_rep = self.f.bias.reshape(1, 1, -1)
            next_m = next_m + b_rep           
            
        if self.mode =="full":
            if new_batch:
                self._A_expanded = self._build_A(m)
            A_expanded = self._A_expanded
            P = P.reshape(batch_size, num_nodes*D_in, num_nodes*D_in)
            next_P = A_expanded@P@A_expanded.transpose(1,2)
            next_P = next_P.reshape(batch_size, num_nodes, self.D_out, num_nodes, self.D_out)
        if self.mode =="major_block":
            A = self.f.weight.unsqueeze(0)
            next_P = P.reshape(-1, D_in, D_in)
            next_P = A@next_P@A.transpose(1,2)
            next_P = next_P.reshape(batch_size, num_nodes, self.D_out, self.D_out)
            
        if self.mode=="all_diagonal":
            A = self.f.weight.unsqueeze(0)**2
            next_P = A@P.transpose(2,3).reshape(-1, D_in, 1)
            next_P = next_P.reshape(batch_size, num_nodes, num_nodes, -1).transpose(2,3)
        
        if self.mode =="major_diagonal":
            A = self.f.weight.unsqueeze(0)**2
            next_P = A@P.transpose(1,2)
            next_P = next_P.transpose(1,2).reshape(batch_size, num_nodes, self.D_out)
        return next_m, next_P
     
    def forward(self, x, *args):
        """Monte Carlo forward pass.
        
        Args:
            x: Input with shape (batch_size, num_nodes, D_in).
            
        Returns:
            y: Output with shape (batch_size, num_nodes, D_out).
        """
        batch_size, num_nodes, D_in = x.shape
        
        x = x.reshape(batch_size*num_nodes, D_in)
        x = self.f(x)
        x = x.reshape(batch_size, num_nodes, self.D_out)
        return x
    
    
class MeanAggregator(torch.nn.Module):
    """Mean aggregation.

    Args:
        concat: If 'True' concat state and message.
        mode: Covariance approximation mode.
    """
    def __init__(self, mode="full", concat=False):
        super().__init__()
        self.eligable_modes = ["full", 
                               "major_block",
                               "all_diagonal",
                               "major_diagonal"]
        assert mode in self.eligable_modes, "wrong mode"
        self.concat = concat
        self.mode=mode
            
    def _block_expand(self, A, num_features):
        """Create augmented adjacency matrix."""
        device = A.device
        batch_size, num_nodes, _ = A.shape
        A = A.repeat_interleave(repeats=num_features, dim=1)
        A = A.repeat_interleave(repeats=num_features, dim=2)
       
        I = torch.eye(num_features).repeat(repeats=(num_nodes, num_nodes)).to(device)
        return A*I
    
    def _build_A(self, m, A):
        """Concat augmented adjacency with identity if `concat` is `True`."""
        batch_size, num_nodes, num_features = m.shape
        device = m.device
        
        A_expanded = self._block_expand(A, num_features)
        num_neighbours = A_expanded.sum(2).clamp(1).unsqueeze(-1)
        A_expanded = A_expanded/num_neighbours
        
        if self.concat:
            I = torch.eye(num_nodes*num_features)
            I = I.unsqueeze(0).repeat(batch_size, 1, 1)
            I = I.to(device)
            A_expanded = torch.cat((I, A_expanded), 1)
            
            # sort entries
            blck_1 = A_expanded[:, :num_nodes*num_features].reshape(batch_size, 
                                                                    num_nodes, 
                                                                    num_features, 
                                                                    -1)
            blck_2 = A_expanded[:, num_nodes*num_features:].reshape(batch_size, 
                                                                    num_nodes, 
                                                                    num_features, 
                                                                    -1)
            A_expanded = torch.cat((blck_1, blck_2), 2).reshape(A_expanded.shape)
        return A_expanded
        
    def jac(self, m, A, new_batch=True, *args):
        """Return the Jacobian.
        
        Args:
            m: Mean with shape (batch_size, num_nodes, D).
            A: Adjacency matrix with shape (batch_size, num_nodes, num_nodes).
            new_batch: If `True` recalculate the augmented adjacency matrix.
                
        Returns:
            J: Jacobian shape `(batch_size, num_nodes*D_out, num_nodes*D)`,
                    
        `D_out` is `2*D` if `concat` is `True`, `D` otherwise.
        """
        if new_batch:
            self._jac = self._build_A(m, A)
        jac = self._jac
        return jac
    
    def next_moments(self, m, P, A, new_batch=True, *args):
        """Calculate the next moments.
        
        Args:
            m: Mean with shape (batch_size, num_nodes, D).
            P: Covariance with shape 
                if `mode` is `full`: 
                    (batch_size, num_nodes , D, num_nodes, D),
                if `mode` is `major_diagonal`: 
                    (batch_size, num_nodes, D),
                if `mode` is `major_blocks`:
                    (batch_size, num_nodes, D, D),
                if `mode` is `all_diagonal`:
                    (batch_size, num_nodes, D, num_nodes).
            A: Adjacency matrix with shape (batch_size, num_nodes, num_nodes).
            new_batch: If `True` recalculate the augmented adjacency matrix.
                
        Returns:
            next_m: Mean with shape (batch_size, num_nodes, D_out).
            next_P: Covariance with shape 
                if `mode` is `full`: 
                    (batch_size, num_nodes , D_out, num_nodes, D_out),
                if `mode` is `major_diagonal`: 
                    (batch_size, num_nodes, D_out),
                if `mode` is `major_blocks`:
                    (batch_size, num_nodes, D_out, D_out),
                if `mode` is `all_diagonal`:
                    (batch_size, num_nodes, D_out, num_nodes).
                    
        `D_out` is `2*D` if `concat` is `True`, `D` otherwise.
        """
        device = m.device
        
        batch_size, num_nodes, D_in = m.reshape(*A.shape[:2], -1).shape
        D_out = D_in
        if self.concat:
            D_out *= 2
            
        num_neighbours = A.sum(2).clamp(1).unsqueeze(-1)
        A_norm = A/num_neighbours    
        next_m = torch.cat((m, A_norm@m), 2)
        
        if self.mode=="full":
            if new_batch:
                self._A_expanded = self._build_A(m, A)
            A_expanded = self._A_expanded
            P = P.reshape(batch_size, num_nodes*D_in, num_nodes*D_in)
            next_P = A_expanded@P@A_expanded.transpose(1,2)  
            next_P = next_P.reshape(batch_size, num_nodes, D_out, num_nodes, D_out)
        
        if self.mode=="major_diagonal":
            next_P = (A_norm**2)@P
            if self.concat:
                next_P = torch.cat((P, next_P),2)
                
        if self.mode=="major_block":
            A_norm = A_norm.repeat_interleave(D_in**2, 0)
            next_P = P.transpose(1,3).reshape(-1, num_nodes, 1)
            next_P = (A_norm**2)@next_P
            next_P = next_P.reshape(batch_size, D_in, D_in, num_nodes).transpose(1,3)
            if self.concat:
                P_zero = torch.zeros(size=next_P.shape).to(device)
                next_P = torch.cat((torch.cat((P, P_zero), 3), 
                                    torch.cat((P_zero, next_P), 3)), 2)
            
        if self.mode =="all_diagonal":
            next_P = P.transpose(1,2).reshape(-1, num_nodes, num_nodes)
            A_norm = A_norm.repeat_interleave(D_in, 0)
            next_P = A_norm@next_P@A_norm.transpose(1,2)
            next_P = next_P.reshape(batch_size, -1, num_nodes, num_nodes).transpose(1,2)
            if self.concat:
                next_P = torch.cat((P, next_P),2)
            
        return next_m, next_P
        
    def forward(self, x, A):
        """Monte Carlo forward pass.
        
        Args:
            x: Input with shape (batch_size, num_nodes, D).
            A: Adjacency matrix with shape (batch_size, num_nodes, num_nodes).
            
        Returns:
            y: Output with shape (batch_size, num_nodes, D_out).
            
        `D_out` is `2*D` if `concat` is `True`, `D` otherwise.
        """
        num_neighbours = A.sum(2).clamp(1).unsqueeze(-1)
        
        A = A/num_neighbours
        messages = A@x
        if self.concat:
            return torch.cat((x, messages), 2)
        else:
            return messages
    
        
class ReLU(torch.nn.Module):
    """ReLU nonlinearity.

    Args:
        mode: Covariance approximation mode.
    """
    def __init__(self, mode="full"):
        super().__init__()
        
        self.eligable_modes = ["full", 
                               "major_block",
                               "all_diagonal",
                               "major_diagonal"]
        assert mode in self.eligable_modes, "wrong mode"
        self.mode=mode
        
        self._activation = torch.nn.ReLU()
        self._heaviside = Heaviside(mode=mode)
        
        # constants
        self.register_buffer('_eps', torch.FloatTensor([1e-5]))
        self.register_buffer('_one', torch.FloatTensor([1.0]))
        self.register_buffer('_one_ovr_sqrt2pi', torch.FloatTensor([1.0 / np.sqrt(2.0 * np.pi)]))
        self.register_buffer('_one_ovr_sqrt2', torch.FloatTensor([1.0 / np.sqrt(2.0)]))
        self.register_buffer('_one_ovr_2', torch.FloatTensor([1.0/2.0]))
        self.register_buffer('_two', torch.FloatTensor([2.0]))
        self.register_buffer('_twopi', torch.FloatTensor([2.0 * np.pi]))
        
               
    def _standard_gaussian(self, x):
        """
        line 17-18 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        return self._one_ovr_sqrt2pi * torch.exp(-x*x * self._one_ovr_2)

    def _gaussian_cdf(self, x):
        """
        line 20-21 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        return self._one_ovr_2 * (self._one + torch.erf(x * self._one_ovr_sqrt2))

    def _softrelu(self, x):
        """
        line 23-24 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        return self._standard_gaussian(x) + x * self._gaussian_cdf(x)
      
    def _g(self, rho, mu1, mu2):
        """
        line 26-36 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        one_plus_sqrt_one_minus_rho_sqr = (self._one + torch.sqrt(self._one - rho*rho))
        a = torch.asin(rho) - rho / one_plus_sqrt_one_minus_rho_sqr
        safe_a = torch.abs(a) + self._eps
        safe_rho = torch.abs(rho) + self._eps

        A = a / self._twopi
        sxx = safe_a * one_plus_sqrt_one_minus_rho_sqr / safe_rho
        one_ovr_sxy = (torch.asin(rho) - rho) / (safe_a * safe_rho)
    
        return A * torch.exp(-(mu1*mu1 + mu2*mu2) / (self._two * sxx) + one_ovr_sxy * mu1 * mu2)
    
    def _delta(self, rho, mu1, mu2):
        """
        line 38-39 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        return self._gaussian_cdf(mu1) * self._gaussian_cdf(mu2) + self._g(rho, mu1, mu2)
    
    def _relu_full_covariance(self, m, P, mu, P_diag):
        """
        line 39-47 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_layers.py
        """
        batch_size, num_nodes, D_in, _, _ = P.shape
        D_total = num_nodes*D_in
        P = P.reshape(batch_size, D_total, D_total)
        mu1 = mu.unsqueeze(-1)
        mu2 = mu1.permute(0,2,1)

        s11s22 = P_diag.unsqueeze(2) *  P_diag.unsqueeze(1)
        rho = P / (torch.sqrt(s11s22) + self._eps)
        rho = torch.clamp(rho, -1/(1+1e-5), 1/(1+1e-5))
        return P * self._delta(rho, mu1, mu2)   
    
    def _relu_block_covariance(self, m, P, mu, P_diag):
        batch_size, num_nodes, D_in, _ = P.shape
        D_total = num_nodes*D_in
        
        mu = mu.reshape(batch_size*num_nodes, D_in)
        P = P.reshape(batch_size*num_nodes, D_in, D_in)
        P_diag = P_diag.reshape(batch_size*num_nodes, D_in)
        
        mu1 = mu.unsqueeze(-1)
        mu2 = mu1.permute(0,2,1)

        s11s22 = P_diag.unsqueeze(2) *  P_diag.unsqueeze(1)
        rho = P / (torch.sqrt(s11s22) + self._eps)
        rho = torch.clamp(rho, -1/(1+1e-5), 1/(1+1e-5))

        return P * self._delta(rho, mu1, mu2) 
    
    def _relu_all_diag_covariance(self, m, P, mu, P_diag):
        batch_size, num_nodes, D_in, _ = P.shape
        
        P = P.transpose(1,2).reshape(batch_size*D_in, num_nodes, num_nodes)
        P_diag = P_diag.reshape(batch_size, num_nodes, D_in)
        P_diag = P_diag.transpose(1,2).reshape(batch_size*D_in, num_nodes) 
        mu = mu.reshape(batch_size, num_nodes, D_in).transpose(1,2).reshape(batch_size*D_in, num_nodes)
        
        mu1 = mu.unsqueeze(-1)
        mu2 = mu1.permute(0,2,1)

        s11s22 = P_diag.unsqueeze(2) *  P_diag.unsqueeze(1)
        rho = P / (torch.sqrt(s11s22) + self._eps)
        rho = torch.clamp(rho, -1/(1+1e-5), 1/(1+1e-5))
        return P * self._delta(rho, mu1, mu2)   
    
    def _relu_major_diag_covariance(self, m, P, mu, P_diag):
        """
        line 63-65 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_layers.py
        """
        batch_size, num_nodes, D_in = P.shape
        D_total = num_nodes*D_in
        P = P.reshape(batch_size, D_total)
        
        pdf = self._standard_gaussian(mu) 
        cdf = self._gaussian_cdf(mu)
        softrelu = pdf + mu*cdf
        
        P = P * (cdf + mu*softrelu - (softrelu**2))
        return torch.nn.functional.relu(P).clamp(0)
    
    def jac(self, m, P, *args):
        """Return the Jacobian.
        
        Args:
            m: Mean with shape (batch_size, num_nodes, D).
            P: Covariance with shape 
                if `mode` is `full`: 
                    (batch_size, num_nodes , D, num_nodes, D),
                if `mode` is `major_diagonal`: 
                    (batch_size, num_nodes, D),
                if `mode` is `major_blocks`:
                    (batch_size, num_nodes, D, D),
                if `mode` is `all_diagonal`:
                    (batch_size, num_nodes, D, num_nodes).  
                    
        Returns:
            J: Jacobian shape `(batch_size, num_nodes*D, num_nodes*D)`,       
        """
        return torch.diag_embed(self._heaviside.next_mean(m, P))
    
    def _get_P_diag(self, P):
        """Extract diagonal from covariance matrix."""
        if self.mode =="full":
            batch_size, num_nodes, D_in, _, _ = P.shape
            D_total = num_nodes * D_in
            P_diag = torch.diagonal(P.reshape(batch_size, D_total, D_total), offset=0, dim1=1, dim2=2)
        if self.mode=="major_diagonal":
            batch_size, num_nodes, D_in = P.shape
            P_diag = P.reshape(batch_size, -1)
        if self.mode=="major_block":
            batch_size, num_nodes, D_in, _ = P.shape
            P_diag = torch.diagonal(P, 0, 2, 3).reshape(batch_size, -1)
        if self.mode=="all_diagonal":
            batch_size, num_nodes, D_in, _ = P.shape
            P_diag = P[:,range(num_nodes),:,range(num_nodes)].transpose(0,1).reshape(batch_size, -1)
        return P_diag
    
    def next_moments(self, m, P, *args):     
        """Calculate the next moments.
        
        Args:
            m: Mean with shape (batch_size, num_nodes, D).
            P: Covariance with shape 
                if `mode` is `full`: 
                    (batch_size, num_nodes , D, num_nodes, D),
                if `mode` is `major_diagonal`: 
                    (batch_size, num_nodes, D),
                if `mode` is `major_blocks`:
                    (batch_size, num_nodes, D, D),
                if `mode` is `all_diagonal`:
                    (batch_size, num_nodes, D, num_nodes).
                
        Returns:
            next_m: Mean with shape (batch_size, num_nodes, D).
            next_P: Covariance with shape 
                if `mode` is `full`: 
                    (batch_size, num_nodes , D, num_nodes, D),
                if `mode` is `major_diagonal`: 
                    (batch_size, num_nodes, D),
                if `mode` is `major_blocks`:
                    (batch_size, num_nodes, D, D),
                if `mode` is `all_diagonal`:
                    (batch_size, num_nodes, D, num_nodes).
        """
        batch_size, num_nodes, D_in = m.shape
        D_total = num_nodes*D_in
        m = m.reshape(batch_size, D_total)
        
        P_diag = self._get_P_diag(P)
        P_diag_sqrt = torch.sqrt(P_diag)
        mu = m / (P_diag_sqrt + self._eps)
        m_next = P_diag_sqrt * self._softrelu(mu)
            
        if self.mode =="full":
            P_next = self._relu_full_covariance(m_next, P, mu, P_diag)
            P_next = P_next.reshape(batch_size, num_nodes, D_in, num_nodes, D_in)
        
        if self.mode =="major_diagonal":
            P_next = self._relu_major_diag_covariance(m_next, P, mu, P_diag)
            P_next = P_next.reshape(batch_size, num_nodes, D_in)
        
        if self.mode =="major_block":
            P_next = self._relu_block_covariance(m_next, P, mu, P_diag)
            P_next = P_next.reshape(batch_size, num_nodes, D_in, D_in)
        
        if self.mode=="all_diagonal":
            P_next = self._relu_all_diag_covariance(m_next, P, mu, P_diag)
            P_next = P_next.reshape(batch_size, D_in, num_nodes, num_nodes)
            P_next = P_next.transpose(1,2)
        return m_next.reshape(batch_size, num_nodes, D_in), P_next
        
    def forward(self, x, *args):
        """Monte Carlo forward pass.
        
        Args:
            x: Input with shape (batch_size, num_nodes, D).
            
        Returns:
            y: Output with shape (batch_size, num_nodes, D).
        """
        if x.dim()==3:
            batch_size, num_nodes, num_features = x.shape
            return self._activation(x.reshape(batch_size, -1)).reshape(x.shape)
        else:
            return self._activation(x)
    
        
class Heaviside(torch.nn.Module):
    """Heaviside nonlinearity.

    Args:
        mode: Covariance approximation mode.
    """
    def __init__(self, mode):
        super().__init__()
        
        self.eligable_modes = ["full", 
                               "major_block",
                               "all_diagonal",
                               "major_diagonal"]
        assert mode in self.eligable_modes, "wrong mode"
        self.mode=mode
        
        self.register_buffer('_eps', torch.FloatTensor([1e-5]))
        self.register_buffer('_one', torch.FloatTensor([1.0]))
        self.register_buffer('_one_ovr_sqrt2', torch.FloatTensor([1.0 / np.sqrt(2.0)]))
        self.register_buffer('_one_ovr_2', torch.FloatTensor([1.0/2.0]))
        self.register_buffer('_two', torch.FloatTensor([2.0]))
        self.register_buffer('_twopi', torch.FloatTensor([2.0 * np.pi]))
    
    def _gaussian_cdf(self, x):
        """
        line 20-21 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        return self._one_ovr_2 * (self._one + torch.erf(x * self._one_ovr_sqrt2))
    
    def get_P_diag(self, P):
        if self.mode =="full":
            batch_size, num_nodes, D_in, _, _ = P.shape
            D_total = num_nodes * D_in
            P_diag = torch.diagonal(P.reshape(batch_size, D_total, D_total), offset=0, dim1=1, dim2=2)
        if self.mode=="major_diagonal":
            batch_size, num_nodes, D_in = P.shape
            P_diag = P.reshape(batch_size, -1)
        if self.mode=="major_block":
            batch_size, num_nodes, D_in, _ = P.shape
            P_diag = torch.diagonal(P, 0, 2, 3).reshape(batch_size, -1)
        if self.mode=="all_diagonal":
            batch_size, num_nodes, D_in, _ = P.shape
            P_diag = P[:,range(num_nodes),:,range(num_nodes)].transpose(0,1).reshape(batch_size, -1)
        return P_diag
    
    def next_mean(self, m, P):
        """Calculate the next mean.
        
        Args:
            m: Mean with shape (batch_size, num_nodes, D).
            P: Covariance with shape 
                if `mode` is `full`: 
                    (batch_size, num_nodes , D, num_nodes, D),
                if `mode` is `major_diagonal`: 
                    (batch_size, num_nodes, D),
                if `mode` is `major_blocks`:
                    (batch_size, num_nodes, D, D),
                if `mode` is `all_diagonal`:
                    (batch_size, num_nodes, D, num_nodes).
                
        Returns:
            next_m: Mean with shape (batch_size, num_nodes, D).
        """
        batch_size, num_nodes, D_in = m.shape
        D_total = num_nodes*D_in
        m = m.reshape(batch_size, D_total)
        
        P_diag = self.get_P_diag(P)
        P_diag_sqrt = torch.sqrt(P_diag)
        mu = m / (P_diag_sqrt + self._eps)
        
        m = self._gaussian_cdf(mu)
        return m
        
    def forward(self, x):
        raise Exception("No Forward Mode") 
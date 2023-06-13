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

class BaseGDSSM(torch.nn.Module):
    """Base class of a GDSSSM. 

    Args:
        mode: Covariance approximation mode.
        mask: Instance of 'layers.Mask'.
        dt: time step
    """
    def __init__(self, mode, mask=None, dt=.1):
        super().__init__()
        self.register_buffer('dt',torch.FloatTensor([dt]))
        self.mode = mode
        self.mask=mask

    def drift(self, x, A):
        raise NotImplementedError("Drift not implemented.")

    def drift_moments(self, m, P, A, new_batch=True):
        raise NotImplementedError("Drift-Moment not implemented.")
        
    def diffusion(self, x, A):
        raise NotImplementedError("Diffusion not implemented.")
        
    def diffusion_moments(self, m, P, A=None, new_batch=True):
        raise NotImplementedError("Diffusion-Moment not implemented.")
        
    def expected_gradient(self):
        return self.exp_jac
      
    def transpose_Pxf(self, P):
        """Transpose covariance matrix..
        
        Args:
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
            P_T: Covariance with shape 
                if `mode` is `full`: 
                    (batch_size, num_nodes , D_out, num_nodes, D_out),
                if `mode` is `major_diagonal`: 
                    (batch_size, num_nodes, D_out),
                if `mode` is `major_blocks`:
                    (batch_size, num_nodes, D_out, D_out),
                if `mode` is `all_diagonal`:
                    (batch_size, num_nodes, D_out, num_nodes).
        """
        
        if self.mode=="full":
            batch_size, num_nodes, D, *_ = P.shape
            D_tot = num_nodes*D
            P_T = P.reshape(batch_size, D_tot, D_tot).transpose(1,2)
            P_T = P_T.reshape(P.shape)
        if self.mode=="major_block":
            P_T = P.transpose(2,3)
        if self.mode =="all_diagonal":
            P_T = P.transpose(1,3)
        if self.mode =="major_diagonal":
            P_T=P
        return P_T
    
    def get_P_diag(self,P):
        """Extract diagonal from covariance matrix.
        
        Args:
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
            P_diag: Covariance with shape (batch_size, num_nodes, D).
        """
        if self.mode=="major_diagonal":
            P_diag = P
        if self.mode=="all_diagonal":
            P_diag = torch.diagonal(P,0, 1, 3).transpose(1,2)
        if self.mode=="full":
            batch_size, num_nodes, D, *_ = P.shape
            D_tot = num_nodes*D
            P_diag = torch.diagonal(P.reshape(batch_size, D_tot, D_tot),0,1,2)
            P_diag = P_diag.reshape(batch_size, num_nodes, D)
        if self.mode=="major_block":
            batch_size, num_nodes, D, *_ = P.shape
            P_diag = torch.diagonal(P,0,2,3)
            P_diag = P_diag.reshape(batch_size, num_nodes, D)
        return P_diag  
    
    def embed_P_diag(self,P_diag):
        """Embed diagonal covariance into some other covariance shape.
        
        Args:
            P_diag: Covariance with shape (batch_size, num_nodes, D).
                          
        Returns:
            P: Covariance with shape 
                if `mode` is `full`: 
                    (batch_size, num_nodes , D, num_nodes, D),
                if `mode` is `major_diagonal`: 
                    (batch_size, num_nodes, D),
                if `mode` is `major_blocks`:
                    (batch_size, num_nodes, D, D),
                if `mode` is `all_diagonal`:
                    (batch_size, num_nodes, D, num_nodes).        
        """
        batch_size, num_nodes, D = P_diag.shape
        if self.mode =="full":
            P = torch.diag_embed(P_diag.reshape(batch_size, -1))
            P = P.reshape(batch_size, num_nodes, D, num_nodes, D)
        if self.mode=="major_block":
            P = torch.diag_embed(P_diag)
        if self.mode=="all_diagonal":
            P = torch.diag_embed(P_diag.transpose(1,2)).transpose(1,2)
        if self.mode=="major_diagonal":
            P = P_diag
        return P
        
    def diffusion_central_moment(self, m, P, A=None, new_batch=True):
        """Central moment of diffusion.
        
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
            new_batch: If `True` recalculate the augmented weight and adjacency matrix.
                
        Returns:
            P_central: Central moment with shape with shape (batch_size, num_nodes, D).
        """
        batch_size, num_nodes, D = m.shape
        m, P = self.diffusion_moments(m, P, A, new_batch)
        P_diag = self.get_P_diag(P)
        m_sqr = torch.diagonal(m.reshape(batch_size,-1,1)@m.reshape(batch_size,1,-1),0,1,2)
        m_sqr = m_sqr.reshape(batch_size, num_nodes, D)
        P_central = P_diag + m_sqr
        return P_central
    
    def next_moments(self, m, P, A=None, new_batch=True):
        """Central moment of diffusion.
        
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
            new_batch: If `True` recalculate the augmented weight and adjacency matrix.
                
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
        
        f_m, P_ff = self.drift_moments(m, P, A, new_batch)
        f_m = f_m*self.dt
        P_ff = P_ff*(self.dt**2) # Cov(f)                           
        L_P_central = self.diffusion_central_moment(m, P, A, new_batch)*self.dt # E[LL^T]

        J = self.exp_jac # Expected Jacobian
        P_xf = self.mask.JP_product(J, P) # Cov(x,f), Eq. 16
        
        P = P +  self.embed_P_diag(L_P_central) # Eq. 23
        P = P + P_ff + P_xf*(self.dt) + self.transpose_Pxf(P_xf)*(self.dt)
        m = m + f_m
        return m, P
    
    def forward(self, x, A=None):
        """Monte Carlo forward pass.
        
        Args:
            x: Input with shape (batch_size, num_nodes, D).
            A: Adjacency matrix with shape (batch_size, num_nodes, num_nodes).
            
        Returns:
            y: Output with shape (batch_size, num_nodes, D).
        """
        device = x.device
        noise = torch.randn(*x.shape).to(device)*torch.sqrt(self.dt)
        x = x + self.drift(x, A)*self.dt + self.diffusion(x, A)*noise
        return x


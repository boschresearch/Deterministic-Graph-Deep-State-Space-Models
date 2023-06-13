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

import sys; sys.path.insert(0, '../')
import os
import argparse
import torch
from torch.utils.data import  DataLoader

from gdssm.layers import Mask
from gdssm.networks_round import Encoder, Dynamics, Decoder
from gdssm.utils import wrap_mP, NLL, RounDDataset

parser = argparse.ArgumentParser('Train rounD')
parser.add_argument('--sparsity_mode', default="full", help="Covariance approximation")
parser.add_argument('--num_modes', default=2, help="Number of modes in the GMM.")
parser.add_argument('--D_e', default=4, help="Latent dimensonality.")
parser.add_argument('--batch_size', default=4, help="Batch size.")
parser.add_argument('--checkpoint', default=True, help="Checkpoint after each epoch.")
parser.add_argument('--verbose', default=True, help="Show loss at each iteration.")
parser.add_argument('--output_path', help="Where to store weights.")
parser.add_argument('--path_to_data', help="Where to load data from.")
args = parser.parse_args()

if __name__ == "__main__":
    sparsity_mode = args.sparsity_mode
    num_modes = args.num_modes
    D_e = args.D_e
    batch_size = args.batch_size 
    checkpoint =  args.checkpoint
    verbose = args.verbose 
    output_path = args.output_path
    path_to_data=args.path_to_data
    
    # check device
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = "cuda:0"
    else:
        device = "cpu"


    scenario_Ids_train = list(range(2,20))
    rounD_train = RounDDataset(path_A=path_to_data, path_D=path_to_data, scenario_Ids=scenario_Ids_train)

    # normalizing
    flattened_D_train = rounD_train.D.reshape(-1,2)
    mask_train = (flattened_D_train[:,0]!=0) & (flattened_D_train[:,1]!=0)
    mean = flattened_D_train[mask_train].mean(0).unsqueeze(0)
    std =  flattened_D_train[mask_train].std(0).unsqueeze(0)
    flattened_D_train_normalized = torch.clone(flattened_D_train)
    flattened_D_train_normalized[mask_train] = (flattened_D_train[mask_train] - mean)/std
    D_train_normalized = flattened_D_train_normalized.reshape(rounD_train.D.shape)
    rounD_train.D = D_train_normalized

    # init networks
    enc = Encoder(D_e=D_e, num_modes=num_modes).to(device)
    dyn = Dynamics(D_x=D_e, dt=1, mode=sparsity_mode).to(device)
    dec = Decoder(D_e=D_e, mode=sparsity_mode).to(device)

    params = list(enc.parameters()) + list(dyn.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    history = []
    for epoch in range(500):
        trainDataloader = DataLoader(rounD_train, batch_size=batch_size, shuffle=True)
        it_trainDataloader = iter(trainDataloader)
        for counter, (hist_batch, fut_batch, A_batch) in enumerate(it_trainDataloader):
            try:
                opt.zero_grad()
                max_load = int((A_batch.sum(1)>0).sum(1).max().numpy())
                max_load = max(1, max_load)

                mask = Mask(mask_type=sparsity_mode, D=D_e, num_nodes=max_load)
                dyn.mask=mask

                hist_batch = hist_batch[:,:max_load]
                fut_batch = fut_batch[:,:max_load]
                A_batch = A_batch[:, :max_load, :max_load]

                hist_batch = hist_batch.to(device)
                fut_batch = fut_batch.to(device)
                A_batch = A_batch.to(device)

                prod = (fut_batch[:, :, :, 0] == 0).float() * (fut_batch[:, :, :, 1] == 0).float()
                mask_batch = torch.cumprod(1 - prod, dim=2)    
                batch_size, num_nodes, horizon, num_feat = fut_batch.shape   

                m_enc, P_enc, weights = enc(hist_batch, A_batch)
                m_enc = m_enc.reshape(batch_size*num_modes, num_nodes, D_e)
                P_enc = P_enc.reshape(batch_size*num_modes, num_nodes, D_e, num_nodes, D_e)
                P_enc = mask.filter_P(P_enc)

                A_batch_modes = A_batch.unsqueeze(1).repeat(1,num_modes, 1, 1)
                A_batch_modes = A_batch_modes.reshape(batch_size*num_modes, num_nodes, num_nodes)

                m_dec, P_dec = [], []
                for h in range(horizon):
                    if h==0:
                        new_batch=True
                    else:
                        new_batch=False
                    m_enc, P_enc = dyn.next_moments(m_enc, P_enc, A_batch_modes)       
                    m_dec_, P_dec_ = dec.next_moments(m_enc, P_enc)
                    m_dec.append(m_dec_), P_dec.append(P_dec_)

                m_dec = torch.stack(m_dec)
                P_dec = torch.stack(P_dec)

                # target shape: batch_size, num_modes, horizon, num_nodes, num_feat
                fut_batch_modes = fut_batch.unsqueeze(1).repeat(1, num_modes, 1, 1,1).transpose(2,3)
                m_wrapped, P_wrapped = wrap_mP(m_dec, P_dec, sparsity_mode)
                m_wrapped = m_wrapped.reshape(batch_size, num_modes, horizon, num_nodes, num_feat)
                P_wrapped = P_wrapped.reshape(batch_size, num_modes, horizon, num_nodes, num_feat, num_feat)

                nll_mode = NLL(fut_batch_modes.reshape(-1, num_feat),
                                        m_wrapped.reshape(-1, num_feat),
                                        P_wrapped.reshape(-1, num_feat, num_feat),
                                        jitter=1e-3)
                nll_mode = nll_mode.reshape(batch_size, num_modes, horizon, num_nodes)
                weighted_likelihood = (torch.exp(-nll_mode)*weights.unsqueeze(-1).unsqueeze(-1)).sum(1)
                weighted_nll = -torch.log(weighted_likelihood+1e-6)
                mask_batch = mask_batch.transpose(1,2)
                masked_weighted_nll = torch.sum((weighted_nll*mask_batch)/torch.sum(mask_batch))

                if not torch.isnan(masked_weighted_nll).sum()>0:
                    masked_weighted_nll.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1)
                    opt.step()
                    history.append(masked_weighted_nll.detach().cpu().numpy())
                    if verbose:
                        print("finished iter {} nll: {}".format(counter, history[-1]))

            except:
                counter_str = str(counter).zfill(6)
                epoch_str = str(epoch).zfill(6)
                print("Error at epoch {} at iter {}".format(epoch_str, counter_str))
        
        if checkpoint:
            epoch_str = str(epoch).zfill(6)
            if verbose:
                print("save after epoch {} || nll: {}".format(epoch_str, history[-1]))
                
            path_enc = os.path.join(output_path, "enc_{}_modes_{}_epoch_{}".format(sparsity_mode, num_modes, epoch_str))
            path_dyn = os.path.join(output_path, "dyn_{}_modes_{}_epoch_{}".format(sparsity_mode, num_modes, epoch_str))
            path_dec = os.path.join(output_path, "dec_{}_modes_{}_epoch_{}".format(sparsity_mode, num_modes, epoch_str))
            torch.save(enc.state_dict(), path_enc)
            torch.save(dyn.state_dict(), path_dyn)
            torch.save(dec.state_dict(), path_dec)
            
    # saver after training
    path_enc = os.path.join(output_path, "enc_{}_modes_{}".format(sparsity_mode, num_modes))
    path_dyn = os.path.join(output_path, "dyn_{}_modes_{}".format(sparsity_mode, num_modes))
    path_dec = os.path.join(output_path, "dec_{}_modes_{}".format(sparsity_mode, num_modes))
    torch.save(enc.state_dict(), path_enc)
    torch.save(dyn.state_dict(), path_dyn)
    torch.save(dec.state_dict(), path_dec)
    
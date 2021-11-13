from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full_1 = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                                 2*self.atom_fea_len)
        self.fc_full_2 = nn.Linear(2*self.atom_fea_len,self.atom_fea_len)
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()
        self.relu = nn.ReLU()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        M1=int(M/2)
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        atom_in_fea_u = atom_in_fea.unsqueeze(1).expand(N, M1, self.atom_fea_len)
        atom_nbr_fea_1, atom_nbr_fea_2 = atom_nbr_fea.chunk(2,dim=1)
        nbr_fea_1,nbr_fea_2 = nbr_fea.chunk(2,dim=1)
        update_nbr_fea_1 = torch.cat([atom_in_fea_u,atom_nbr_fea_1, nbr_fea_1], dim=2)
        update_nbr_fea_2 = torch.cat([atom_in_fea_u,atom_nbr_fea_2, nbr_fea_2], dim=2)
        total_gated_fea_1 = self.fc_full_1(update_nbr_fea_1)
        total_gated_fea_2 = self.fc_full_1(update_nbr_fea_2)
        total_gated_fea_1 = self.relu(total_gated_fea_1)
        total_gated_fea_2 = self.softplus1(total_gated_fea_2)
        total_gated_fea = torch.cat([total_gated_fea_1,total_gated_fea_2], dim=1)
        nbr_sumed = torch.sum(total_gated_fea, dim=1)
        nbr_sumed = self.fc_full_2(nbr_sumed)
        nbr_sumed = self.bn1(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128,n_p=1,
                 classification=False,dropout=0):
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.n_p = n_p
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.ModuleList([nn.Linear(atom_fea_len+4, h_fea_len)\
                for _ in range(self.n_p)])
        self.conv_to_fc_activation = nn.ModuleList([nn.Softplus() for _ in range(self.n_p)])
        self.fc_softplus = nn.Softplus()
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.ModuleList([nn.Linear(h_fea_len, 1) for _ in range(self.n_p)])
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, M1_index, crystal_atom_idx,m2_fea):
        atom_fea = self.embedding(atom_fea)
        atom_fea = self.fc_softplus(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        m1_fea = atom_fea[M1_index,:]
        crys_fea = self.pooling(m1_fea, crystal_atom_idx)
        crys_fea = torch.cat([crys_fea,m2_fea],dim=1)
        crys_features = [self.conv_to_fc[i](self.conv_to_fc_activation[i](crys_fea))\
            for i in range(self.n_p)]
        crys_features = [self.conv_to_fc_activation[i](crys_features[i]) for i in range(self.n_p)]
        processed_features = []
        for i in range(self.n_p):
            out_val = crys_features[i]
            processed_features.append(out_val)
        out = [self.fc_out[i](processed_features[i]) for i in range(self.n_p)]
        out = torch.cat(out, 1)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

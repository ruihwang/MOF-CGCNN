from __future__ import print_function, division

import csv
import functools
import json
import os,sys
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader


def collate_pool(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_m1_index = [], [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    batch_m2_fea = []
    base_idx = 0
    base_m1_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx,m1_index,m2_feature), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        m1_i = m1_index.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        batch_m1_index.append(m1_index + base_idx)
        new_idx = torch.LongTensor(np.arange(m1_i)+base_m1_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        batch_m2_fea.append(m2_feature)
        base_idx += n_i
        base_m1_idx += m1_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(batch_m1_index,dim=0),
            crystal_atom_idx,
            torch.stack(batch_m2_fea,dim=0)),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

class CIFData(Dataset):
    def __init__(self, root_dir,dataset, max_num_nbr=10,metal_max_num_nbr=16, metal_radius=8, radius=6, dmin=0, step=0.2,
                 random_seed=24,pred=False):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.metal_max_num_nbr, self.metal_radius = metal_max_num_nbr, metal_radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        self.id_prop_data = dataset
        self.pred = pred
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        if self.pred == False:
            cif_id = str(self.id_prop_data[idx][0])
            target = [float(x) for x in self.id_prop_data[idx][5:6]]
        else:
            cif_id,target = str(self.id_prop_data[idx][0]),[float(self.id_prop_data[idx][-1])]
        m2_feature = [float(self.id_prop_data[idx][1]),float(self.id_prop_data[idx][2]),float(self.id_prop_data[idx][3]),float(self.id_prop_data[idx][4])]
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id+'.cif'))
        metal_index = []
        metal_number = 0
        atom_fea = []
        for i in range(len(crystal)):
            if crystal[i].specie.is_metal:
                metal_index.append(i)
                metal_number += 1
        if metal_number == 0 :
            print("There is no metal atoms in MOFs check ID "+str(cif_id)+'\n')
            sys.exit(0)
        for i in range(len(crystal)):
            fea = [crystal[i].specie.number,len(crystal.get_neighbors(crystal[i], 1.6))]
            atom_fea.append(fea)
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        metal_neighs =[]
        # metal_neighs_atoms_index
        for i in range(len(crystal)):
            site = crystal[i]
            metal = site.specie.is_metal
            if metal:
                neighs_dists = crystal.get_neighbors(site, self.metal_radius, include_index=True)
                metal_neighs.append(neighs_dists)
        all_metal_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in metal_neighs]
        metal_nbr_idx = []
        M_idx = []
        for nbr in all_metal_nbrs:
            if len(nbr) < self.metal_max_num_nbr:
                warnings.warn('{} not find enough neighbors for metal atoms . '
                             'If it happens frequently, consider increase '
                             'radius.'.format(cif_id))
                metal_nbr_idx.append(list(map(lambda x: x[2], nbr)) +
                                            [0] * (self.metal_max_num_nbr - len(nbr)))
            else:
                metal_nbr_idx.append(list(map(lambda x: x[2],
                        nbr[:self.metal_max_num_nbr])))
        for i in range(len(metal_nbr_idx)):
            assert len(metal_nbr_idx[i]) == self.metal_max_num_nbr
            for j in range(self.metal_max_num_nbr):
                M_idx.append(metal_nbr_idx[i][j])
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))

        M1_idx = list(set(M_idx)|set(metal_index))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        M1_index = torch.LongTensor(M1_idx)
        target = torch.Tensor(target)
        M2_feature = torch.Tensor(m2_feature)
        return (atom_fea, nbr_fea, nbr_fea_idx,M1_index,M2_feature), target, cif_id

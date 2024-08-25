import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from lib.logger import *

@DATASETS.register_module()
class ShapeNetFeat(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        self.sample_points_num = config.npoints
        self.sv_k = config.get('sv_k', 128)

        self.whole = True # whether train on all data

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeNetFeat')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeNetFeat')
        
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeNetFeat')
            lines = test_lines + lines
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('/')[0]
            model_id = line.split('/')[1].split('.')[0]

            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
            })

        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNetFeat')
        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, normal, sv, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        normal = normal[self.permutation[:num]]
        sv = sv[self.permutation[:num]]
        return pc, normal, sv
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        
        sv_path = os.path.join(self.data_root, 'sv_k%d' % self.sv_k, sample['taxonomy_id'], sample['model_id'] + '.npy')
        pc_path = os.path.join(self.data_root, 'pointcloud', sample['taxonomy_id'], sample['model_id'] + '.npy')
        sv = IO.get(sv_path).astype(np.float32)
        pc_data = IO.get(pc_path).astype(np.float32)
        pc = pc_data[:,:3]
        normal = pc_data[:,3:]

        sample_pc, sample_normal, sample_sv = self.random_sample(pc, normal, sv, self.sample_points_num)

        sample_pc = torch.from_numpy(sample_pc).float()
        sample_sv = torch.from_numpy(sample_sv).float()
        sample_normal = torch.from_numpy(sample_normal).float()

        data = {'pc': sample_pc, 'sv': sample_sv, 'normal': sample_normal}
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)

import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pointnet_util import farthest_point_sample, pc_normalize
import json


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)

class ScanNetDataset(Dataset):
    def __init__(self, root='../../data/scannet_scenes/', data_prefix='whole_scene', npoints=8192, split='train'):
        self.npoints = npoints

        self.split = 'train' if split == 'train' else 'val'
        self.data_prefix = data_prefix

        self.filename = os.path.join(root, '%s_%s.txt' % (data_prefix, self.split))
        self.root = root
        
        with open(self.filename, 'r') as f:
            self.scan_names = f.read().splitlines()   
    
    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
 
    def __getitem__(self, index):

        scan_name = self.scan_names[index]

        data = np.load(os.path.join(self.root, self.data_prefix, scan_name + '.npy'))
        pc = data[:,:3]
        ins_label = data[:, 6]
        label = data[:, 7]
           
        pc = pc_normalize(pc)
        
        self.permutation = np.arange(pc.shape[0])
        
        if self.npoints != 0:
            if self.split == 'train':
                np.random.shuffle(self.permutation) 
                pc = pc[self.permutation[:self.npoints]]
                label = label[self.permutation[:self.npoints]]
                ins_label = ins_label[self.permutation[:self.npoints]]
            
        ret = {'pc': pc, 'sem_label': label, 'ins_label': ins_label}
       
        return ret

    def __len__(self):
        return len(self.scan_names)


class ScanNetChunkDataset(Dataset):
    def __init__(self, root='../../data/scannet_scenes/', data_prefix='whole_scene', npoints=8192, split='train'):
        self.npoints = npoints

        self.split = 'train' if split == 'train' else 'val'
        self.data_prefix = data_prefix

        self.filename = os.path.join(root, '%s_%s.txt' % (data_prefix, self.split))
        self.root = root
        
        with open(self.filename, 'r') as f:
            self.scan_names = f.read().splitlines()   

        self.dict_names = {} # scene chunks
        self.new_scan_names = []
        
        for i in self.scan_names:
            scene_name = i.split('_chunk')[0]

            if scene_name not in self.new_scan_names:
                self.new_scan_names.append(scene_name)
            
            if scene_name not in self.dict_names:
                self.dict_names[scene_name] = []
            
            self.dict_names[scene_name].append(i)

        self.scan_names = self.new_scan_names # scene list
    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
 
    def __getitem__(self, index):

        scan_name = self.scan_names[index]
        
        chunks_names = self.dict_names[scan_name]
        
        data = np.load(os.path.join(self.root, 'whole_scene', scan_name + '.npy'))
        
        ret = {}
        ret['num'] = data.shape[0]
        #ret['whole_pc'] = data[:,:3] # should be removed, only for debug
        
        ret['datas'] = {}

        for i in chunks_names:
            data = np.load(os.path.join(self.root, self.data_prefix, i + '.npy'))
            pc = data[:,:3]
            pc = pc_normalize(pc)
            ins_label = data[:, 6]
            label = data[:, 7]
            
            idx = np.loadtxt(os.path.join(self.root, self.data_prefix, i + '.index'))

            ret['datas'][i] = {'pc': pc, 'ins_label': ins_label, 'sem_label': label, 'idx': idx}
        
        return ret

    def __len__(self):
        return len(self.scan_names)


if __name__ == '__main__':
    data = ModelNetDataLoader('modelnet40_normal_resampled/', split='train', uniform=False, normal_channel=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)

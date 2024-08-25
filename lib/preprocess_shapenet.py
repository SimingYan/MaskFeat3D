
import os
import numpy as np
import glob
import argparse
import matplotlib
import matplotlib.cm
from multiprocessing import Pool

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', type=str, default='./data/ShapeNetFeat/pointcloud/')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--k', type=int, default=128)
parser.add_argument('--multiprocessing', action='store_true')

args = parser.parse_args()

points_dtype = np.float16
n_iou_points = 3000000

def compute_covariance(patch):
    centroid = patch.mean(0)
    patch = patch - centroid
    cov_mat = np.matmul(patch.T, patch)
    w, v =  np.linalg.eig(cov_mat)
    w.sort()
    sv = w[0] / (w[0] + w[1] + w[2])
    return sv

def compute_surface_variation(pc_path, save_file, k=128):
    pc = np.load(pc_path)[:,:3]
    print(pc_path)
    sv = []
    for i in range(pc.shape[0]):
        cur_point = pc[i]
        crop_idx = np.argsort(np.sum(np.square(pc - cur_point), 1))[:k]
        knn = pc[crop_idx]
        
        cur_sv = compute_covariance(knn)
        sv.append(cur_sv)

    sv = np.array(sv)
    
    #np.save(save_file, sv)

    return sv

object_list = glob.glob(args.dataset_path + '/*')
object_list.sort()

total_list = []

for o in object_list:
    if 'yaml' in o:
        continue

    instance_list = glob.glob(o+'/*')
    instance_list.sort()

    for i in instance_list:
        output_file = i.replace('pointcloud', 'sv_k%d' % args.k)
 
        total_list.append(i)

process_num = len(total_list) // args.split
print('current idx:', args.idx)
process_list = total_list[args.idx*process_num:(args.idx+1)*process_num]


if args.multiprocessing:
    arg_list = []
    cnt = 0
    for i in process_list:

        output_file = i.replace('pointcloud', 'sv_k%d' % args.k)
        
        if os.path.exists(output_file):
            continue

        parent_path = '/'.join(output_file.split('/')[:-1])
        
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)

        arg_list.append([i, output_file, args.k]) 
        print(output_file)
        cnt += 1

    print(cnt)
    
    p = Pool(49)
    p.starmap(compute_surface_variation, arg_list)
else:
    for i in process_list:
        output_file = i.replace('pointcloud', 'sv_k%d' % args.k)
        
        parent_path = '/'.join(output_file.split('/')[:-1])
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)

        sv = compute_surface_variation(i, output_file, k=args.k)
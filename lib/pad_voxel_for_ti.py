import numpy as np
import MinkowskiEngine as ME
from MinkowskiEngine.utils import sparse_quantize
__all__ = ['pad_voxel_at_strides']
def calc_key(points, bound):
    points = points.astype(np.int64)
    bound = bound.astype(np.int64)
    key = points[:, 0] * bound[1] * bound[2] + points[:, 1] * bound[2] + points[:, 2]
    return key

def calc_unique_pts(pts):
    bound = (np.max(pts, axis=0) + 1).astype(np.int64)
    key = calc_key(pts, bound)
    _, index = np.unique(key, return_index=True)
    unique_pts = pts[index]
    return unique_pts

# pad num == 27:
# pad for 27 neighbors
# pad num == 8:
# pad for octree or tri_itp
def pad_neighbors(points, bound, flag, pad_num=8, stride=1):
    if pad_num == 27:
        direction = np.array([[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1], [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1]]) * stride
    elif pad_num == 8:
        direction = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]) * stride
    neighbors = points.repeat(len(direction), 0).reshape(-1, len(direction), 3) + direction
    neighbors = neighbors.reshape(-1, 3)
    mask = np.prod(neighbors >= 0, axis=1).astype(bool)
    neighbors = neighbors[mask]
    mask = np.prod(neighbors < bound, axis=1).astype(bool)
    neighbors = neighbors[mask]
    key = calc_key(neighbors, bound)
    key, unique_index = np.unique(key, return_index=True)
    neighbors = neighbors[unique_index]
    neighbors = neighbors[flag[key]]
    return neighbors

# pc -> flag
# pc_floor -> new_padded
def pad_voxel(pc, pc_floor, pad_num=8, stride=1):
    num_before_pad = len(pc)

    bound = (np.max(pc, axis=0) + stride + 1).astype(np.int64)

    flag = np.ones(bound[0] * bound[1] * bound[2], dtype=np.bool)
    key = calc_key(pc, bound)
    flag[key] = False

    new_padded = pad_neighbors(pc_floor, bound, flag, pad_num=pad_num, stride=stride)
    
    num_padded = len(new_padded)
    num_after_pad = num_before_pad + num_padded
    #print("stride:", stride, ", pad_num:", pad_num, ":", num_before_pad, '->', num_after_pad)
    return new_padded


def pad_voxel_at_strides(voxel_coords, voxel_feats, voxel_labels, pts, voxel_pad_strides=[], feats_pad_value=0, labels_pad_value=-1):
    for stride in voxel_pad_strides:
        pad_coords_floor_ = np.floor(pts / (stride/2)) * (stride/2)
        pad_coords_floor = sparse_quantize(pad_coords_floor_).numpy()
        pad_num = 8

        if pad_num > 0:
            padded_pc = pad_voxel(voxel_coords.astype(np.int32), pad_coords_floor.astype(np.int32), stride=stride, pad_num=pad_num).astype(np.float32)
            padded_num = len(padded_pc)
            voxel_feats = np.pad(voxel_feats, ((0, padded_num), (0, 0)), 'constant', constant_values=feats_pad_value)
            voxel_labels = np.pad(voxel_labels, (0, padded_num), 'constant', constant_values=labels_pad_value)
            voxel_coords = np.concatenate([voxel_coords, padded_pc], axis=0)

    # if reorder the voxel_coords, inverse_map should also be mapped use reorder_inverse
    voxel_num_after_padded = len(voxel_coords)
    _, reorder_inds, reorder_inverse = sparse_quantize(voxel_coords, return_index=True, return_inverse=True)
    assert(voxel_num_after_padded == len(reorder_inds))        
    voxel_coords = voxel_coords[reorder_inds]
    voxel_feats = voxel_feats[reorder_inds]
    voxel_labels = voxel_labels[reorder_inds]
    return voxel_coords, voxel_feats, voxel_labels
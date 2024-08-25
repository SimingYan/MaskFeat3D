from MinkowskiEngine import SparseTensor
from MinkowskiInterpolation import MinkowskiInterpolationFunction
import unittest
import numpy as np
import torch
import time

def trilinear_interpolation(volume_feats, points):
    device = volume_feats.device
    B, res, _, _, C = volume_feats.shape
    assert points.ndim == 3
    assert points.shape[0] == B
    assert points.shape[2] == 3
    n_point = points.shape[1]

    points = torch.clamp(points, min=0, max=res)
    x, y, z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
    x0, y0, z0 = torch.floor(x), torch.floor(y), torch.floor(z)
    x1, y1, z1 = x0+1, y0+1, z0+1
    
    x0 = torch.clamp(x0, 0, res-1)
    x1 = torch.clamp(x1, 0, res-1)
    y0 = torch.clamp(y0, 0, res-1)
    y1 = torch.clamp(y1, 0, res-1)
    z0 = torch.clamp(z0, 0, res-1)
    z1 = torch.clamp(z1, 0, res-1)

    u, v, w = x-x0, y-y0, z-z0
    u = u.view(B, n_point, 1).repeat(1, 1, C)
    v = v.view(B, n_point, 1).repeat(1, 1, C)
    w = w.view(B, n_point, 1).repeat(1, 1, C)
    
    batch_i = torch.arange(B, device=device).view(B, 1).repeat(1, n_point).long()
    c_000 = volume_feats[batch_i, x0.long(), y0.long(), z0.long(), :]
    c_001 = volume_feats[batch_i, x0.long(), y0.long(), z1.long(), :]
    c_010 = volume_feats[batch_i, x0.long(), y1.long(), z0.long(), :]
    c_011 = volume_feats[batch_i, x0.long(), y1.long(), z1.long(), :]
    c_100 = volume_feats[batch_i, x1.long(), y0.long(), z0.long(), :]
    c_101 = volume_feats[batch_i, x1.long(), y0.long(), z1.long(), :]
    c_110 = volume_feats[batch_i, x1.long(), y1.long(), z0.long(), :]
    c_111 = volume_feats[batch_i, x1.long(), y1.long(), z1.long(), :]

    c_xyz = (1.0-u)*(1.0-v)*(1.0-w)*c_000 + \
            (1.0-u)*(1.0-v)*(w)*c_001 + \
            (1.0-u)*(v)*(1.0-w)*c_010 + \
            (1.0-u)*(v)*(w)*c_011 + \
            (u)*(1.0-v)*(1.0-w)*c_100 + \
            (u)*(1.0-v)*(w)*c_101 + \
            (u)*(v)*(1.0-w)*c_110 + \
            (u)*(v)*(w)*c_111

    # compute gradient
    gd_x = c_100 - c_000 + c_110 - c_010 + c_101 - c_001 + c_111 - c_011
    gd_y = c_010 - c_000 + c_110 - c_100 + c_011 - c_001 + c_111 - c_101
    gd_z = c_001 - c_000 + c_101 - c_100 + c_011 - c_010 + c_111 - c_110
    
    grads = torch.cat([gd_x, gd_y, gd_z], dim=-1).float()
    
    grads = grads / (torch.norm(grads, dim=-1, keepdim=True) + 1e-16)
    
    return c_xyz, grads

def trilinear_interpolation_052(sp_tensor, pointcloud, *args):
  device = sp_tensor.device
  v_num = sp_tensor.F.shape[0]
  p_num = pointcloud.shape[0]
  feats = sp_tensor.F.double()
  pc = pointcloud.double()
  
  with torch.no_grad():
    output3, in_map, out_map, weights = MinkowskiInterpolationFunction().apply(
            feats,
            pc,
            sp_tensor.coordinate_map_key,
            sp_tensor.coordinate_manager,
        )
    mat = torch.sparse.DoubleTensor(torch.stack([out_map.long(), in_map.long()]), weights.double(), torch.Size([p_num, v_num])).to(device)
    norm = torch.sparse.mm(mat, torch.ones(v_num, 1).double().to(device))
  
  output = torch.sparse.mm(mat, feats)
  output = torch.div(output, norm+1e-10).float()
  return output

class TestTilinearInterpolation(unittest.TestCase):
  def test_batch_of_shapes_trilinear(self):
    print(f"{self.__class__.__name__}: test_batch_of_shapes_trilinear")
    D = 3
    # define at coords + 0.5
    coords = torch.IntTensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    bcoords1 = torch.IntTensor(coords.shape[0], 1+D)
    bcoords1[:, 0] = 0
    bcoords1[:, 1:] = coords

    coords = torch.IntTensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    bcoords2 = torch.IntTensor(coords.shape[0], 1+D)
    bcoords2[:, 0] = 1
    bcoords2[:, 1:] = coords
    
    feats1 = torch.arange(coords.shape[0]*2).view(coords.shape[0], 2).float()
    feats2 = torch.arange(coords.shape[0]*2).view(coords.shape[0], 2).float() + 100

    bcoords = torch.cat([bcoords1, bcoords2])
    feats = torch.cat([feats1, feats2])

    perm = torch.randperm(bcoords.shape[0])
    bcoords = bcoords[perm]
    feats = feats[perm]
    X = SparseTensor(feats.cuda(), bcoords.int().cuda())
    
    point_cloud1 = torch.FloatTensor([[0.1, 0.2, 0.3], [0.1, 0.2, 1.3], [1.6, 0.2, 1.3]])
    bpoint_cloud1 = torch.FloatTensor(point_cloud1.shape[0], 1+D)
    bpoint_cloud1[:, 0] = 0
    bpoint_cloud1[:, 1:] = point_cloud1

    point_cloud2 = torch.FloatTensor([[0.6, 0.2, 1.3], [0.6, 1.2, 1.3]])
    bpoint_cloud2 = torch.FloatTensor(point_cloud2.shape[0], 1+D)
    bpoint_cloud2[:, 0] = 1
    bpoint_cloud2[:, 1:] = point_cloud2

    bpoint_cloud = torch.cat([bpoint_cloud1, bpoint_cloud2])
    bpoint_cloud = bpoint_cloud.cuda()

    output = trilinear_interpolation_052(X, bpoint_cloud).cpu()
    standard = torch.FloatTensor([[2.2000,   3.2000],
        [  3.6000,   4.6000],
        [ 10.8000,  11.8000],
        [107.6000, 108.6000],
        [110.8000, 111.8000]])
    error = torch.abs(output-standard)
    assert((error<1e-6).all())

  def test_batch_of_shapes_trilinear_strided(self):
    print(f"{self.__class__.__name__}: test_batch_of_shapes_trilinear")
    D = 3
    # define at coords + 0.5
    coords = torch.IntTensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]) * 2
    bcoords1 = torch.IntTensor(coords.shape[0], 1+D)
    bcoords1[:, 0] = 0
    bcoords1[:, 1:] = coords

    coords = torch.IntTensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]) * 2
    bcoords2 = torch.IntTensor(coords.shape[0], 1+D)
    bcoords2[:, 0] = 1
    bcoords2[:, 1:] = coords
    
    feats1 = torch.arange(coords.shape[0]*2).view(coords.shape[0], 2).float()
    feats2 = torch.arange(coords.shape[0]*2).view(coords.shape[0], 2).float() + 100

    bcoords = torch.cat([bcoords1, bcoords2])
    feats = torch.cat([feats1, feats2])

    perm = torch.randperm(bcoords.shape[0])
    bcoords = bcoords[perm]
    feats = feats[perm]
    X = SparseTensor(feats.cuda(), bcoords.int().cuda(), tensor_stride=2)
    
    point_cloud1 = torch.FloatTensor([[0.1, 0.2, 0.3], [0.1, 0.2, 1.3], [1.6, 0.2, 1.3]]) * 2
    bpoint_cloud1 = torch.FloatTensor(point_cloud1.shape[0], 1+D)
    bpoint_cloud1[:, 0] = 0
    bpoint_cloud1[:, 1:] = point_cloud1

    point_cloud2 = torch.FloatTensor([[0.6, 0.2, 1.3], [0.6, 1.2, 1.3]]) * 2
    bpoint_cloud2 = torch.FloatTensor(point_cloud2.shape[0], 1+D)
    bpoint_cloud2[:, 0] = 1
    bpoint_cloud2[:, 1:] = point_cloud2

    bpoint_cloud = torch.cat([bpoint_cloud1, bpoint_cloud2])
    bpoint_cloud = bpoint_cloud.cuda()

    output = trilinear_interpolation_052(X, bpoint_cloud).cpu()
    standard = torch.FloatTensor([[2.2000,   3.2000],
        [  3.6000,   4.6000],
        [ 10.8000,  11.8000],
        [107.6000, 108.6000],
        [110.8000, 111.8000]])
    error = torch.abs(output-standard)
    assert((error<1e-6).all())


if __name__ == '__main__':
  unittest.main()

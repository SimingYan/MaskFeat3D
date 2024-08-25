import torch

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
    return c_xyz


if __name__ == "__main__":
    volume_feats = torch.arange(8).view(2, 2, 2, 1)
    volume_feats = torch.stack([volume_feats, volume_feats*10], dim=0)
    query = torch.Tensor([
        [[-0.5, -0.5, -0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.3, 0.0], [0.4, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [1.5, 1.6, 1.7]],
        [[-0.5, -0.5, -0.5], [0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.2], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [1.5, 1.6, 1.7]],
    ])
    print(trilinear_interpolation(volume_feats, query))

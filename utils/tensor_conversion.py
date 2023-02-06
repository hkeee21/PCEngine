import torch


__all__ = ['to_minkowski_sparseconv_tensor', 'to_spconv_sparseconv_tensor']


def to_minkowski_sparseconv_tensor(x):
    import MinkowskiEngine as ME
    features, coordinates, device = x.F, x.C, x.F.device
    coordinates = coordinates.index_select(1, torch.LongTensor([3, 0, 1, 2]).to(device))
    return ME.SparseTensor(features, coordinates=coordinates, device=device)
    # return ME.SparseTensor(features, coordinates.cpu()).to(device)


def to_spconv_sparseconv_tensor(x, shape=None):
    import spconv
    features, coordinates, device = x.F, x.C, x.F.device
    coordinates = coordinates.index_select(1, torch.LongTensor([3, 0, 1, 2]).to(device))
    # sparse_shape = (torch.max(coordinates[:, 1:], dim=0).values - torch.min(coordinates[:, 1:], dim=0).values + 1).cpu().numpy()
    sparse_shape = (torch.max(coordinates[:, 1:], dim=0).values + 1).cpu().numpy()
    # print(f"sparse shape {sparse_shape}")
    if shape is None:
        return spconv.SparseConvTensor(features, coordinates, sparse_shape, torch.max(coordinates[:, 0]).item() + 1)
    else:
        return spconv.SparseConvTensor(features, coordinates, shape, torch.max(coordinates[:, 0]).item() + 1)

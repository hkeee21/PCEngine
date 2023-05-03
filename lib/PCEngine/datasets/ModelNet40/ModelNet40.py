from ...script.sptensor import spTensor
import torch
import glob
import os
import logging
import open3d as o3d
import numpy as np
from ...script.utils import sparse_quantize

def resample_mesh(mesh_cad, density=1):
    """
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    param mesh_cad: low-polygon triangle mesh in o3d.geometry.TriangleMesh
    param density: density of the point cloud per unit area
    param return_numpy: return numpy format or open3d pointcloud format
    return resampled point cloud

    Reference :
      [1] Barycentric coordinate system
      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    """
    np.random.seed(7)
    faces = np.array(mesh_cad.triangles).astype(int)
    vertices = np.array(mesh_cad.vertices)

    vec_cross = np.cross(
        vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
        vertices[faces[:, 1], :] - vertices[faces[:, 2], :],
    )
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))

    n_samples = (np.sum(face_areas) * density).astype(int)
    # face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Bug fix by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(density * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc : acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]

    P = (
        (1 - np.sqrt(r[:, 0:1])) * A
        + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B
        + np.sqrt(r[:, 0:1]) * r[:, 1:] * C
    )

    return P


class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, item=None, root='AE-dataset/ModelNet40', test_mode='end-to-end'):
        self.files = []
        self.data_objects = []
        self.transform = transform
        if test_mode == 'kernel':
            self.voxel_size = 0.002
        else:
            self.voxel_size = 0.01

        self.root = root
        self.files = []
        if isinstance(item, list):
            for it in item:
                fnames = glob.glob(os.path.join(self.root, it + "/test/*.off"))
                fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])
                self.files = self.files + fnames
        else:
            fnames = glob.glob(os.path.join(self.root, item + "/test/*.off"))
            fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])
            self.files = fnames
        assert len(self.files) > 0, "No file loaded"
        self.density = 4096

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mesh_file = os.path.join(self.root, self.files[idx])
        
        # Load a mesh, over sample, copy, rotate, voxelization
        assert os.path.exists(mesh_file)
        pcd = o3d.io.read_triangle_mesh(mesh_file)
        # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
        vertices = np.asarray(pcd.vertices)
        vmax = vertices.max(0, keepdims=True)
        vmin = vertices.min(0, keepdims=True)
        pcd.vertices = o3d.utility.Vector3dVector(
            (vertices - vmin) / (vmax - vmin).max()
            )

        # Oversample points and copy
        xyz = resample_mesh(pcd, density=self.density)

        if len(xyz) < 2000:
            logging.info(
                f"Skipping {mesh_file}: does not have sufficient CAD sampling density after resampling: {len(xyz)}."
            )
            return None

        if self.transform:
            xyz, feats = self.transform(xyz, feats)

        # Get coords
        xyz -= np.min(xyz, axis=0, keepdims=True)
        coords, inds = sparse_quantize(xyz, voxel_size=self.voxel_size, return_index=True)
        # Use color or other features if available
        feats = np.random.uniform(0, 1, size=(coords.shape[0], 4)) 
        coords = torch.as_tensor(coords, dtype=torch.int)
        feats = torch.as_tensor(feats, dtype=torch.float)
        input = spTensor(coords=coords, feats=feats, 
                        buffer=None, coords_max=None, coords_min=None)

        return {'input': input}

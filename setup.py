import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from version import __version__

sources = [os.path.join('backend', f'pybind_cuda.cpp'), os.path.join('backend', f'spconv.cu'), os.path.join('backend', f'hash.cu')]

setup(
    name='PCEngine',
    version=__version__,
    ext_modules=[
        CUDAExtension('PCEngine.backend',
            sources=sources,
            extra_compile_args = {
                'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
                'nvcc': ['-O3']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [os.path.join('backend', f'pybind_cuda.cpp'), os.path.join('backend', f'spconv.cu')]
extra_compile_args = {
    'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
    'nvcc': ['-O3']
}

setup(
    name='spconvmod',
    ext_modules=[
        CUDAExtension('spconvmod.backend',
            sources, 
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
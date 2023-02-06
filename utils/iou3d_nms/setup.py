from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='iou3d_nms',
    ext_modules=[
        CUDAExtension('iou3d_nms',
                      ['src/iou3d_cpu.cpp', 'src/iou3d_cpu_kernel.cu'],
                      extra_compile_args={
                          'cxx': ['-g'],
                          'nvcc': ['-O2']
                      })
    ],
    cmdclass={'build_ext': BuildExtension},
)

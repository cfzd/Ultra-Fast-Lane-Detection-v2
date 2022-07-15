from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_interp',
    ext_modules=[
        CUDAExtension('my_interp', [
            'my_interp_cuda.cpp',
            'my_interp_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

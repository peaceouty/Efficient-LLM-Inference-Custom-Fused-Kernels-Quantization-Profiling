from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='efficient_llm_kernels',
    version='0.1.0',
    author='Shuai Zhiyu',
    description='Custom CUDA kernels for efficient LLM inference',
    ext_modules=[
        CUDAExtension(
            name='fused_kernels',
            sources=[
                'cuda_kernels/fused_ops.cu',
                'cuda_kernels/bindings.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    '--generate-line-info',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.1.0',
    ],
)

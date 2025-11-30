"""
Accelerators Package
다양한 하드웨어 accelerator를 통일된 인터페이스로 관리
"""

from .base import BaseAccelerator, ProfilingResult
from .cpu_accelerator import CPUAccelerator, create_cpu_accelerator
from .gpu_accelerator import (
    GPUAccelerator, 
    create_gpu_accelerator,
    get_available_gpus,
    create_all_gpu_accelerators
)
from .npu_accelerator import (
    NPUAccelerator,
    create_npu_accelerator,
    create_furiosa_npu,
    create_rebellions_npu
)

__all__ = [
    # Base classes
    "BaseAccelerator",
    "ProfilingResult",
    
    # CPU
    "CPUAccelerator",
    "create_cpu_accelerator",
    
    # GPU
    "GPUAccelerator",
    "create_gpu_accelerator",
    "get_available_gpus",
    "create_all_gpu_accelerators",
    
    # NPU
    "NPUAccelerator",
    "create_npu_accelerator",
    "create_furiosa_npu",
    "create_rebellions_npu",
]

__version__ = "0.2.0"

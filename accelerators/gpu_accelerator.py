"""
GPU (CUDA) Accelerator Implementation
NVIDIA GPU를 사용한 모델 프로파일링
"""

import torch
from .base import BaseAccelerator


class GPUAccelerator(BaseAccelerator):
    """
    GPU (CUDA) Accelerator
    
    특징:
    - CUDA 사용 가능 시에만 동작
    - GPU 메모리 직접 측정
    - 멀티 GPU 지원
    """
    
    def __init__(self, device_id: int = 0):
        """
        Args:
            device_id: GPU ID (0, 1, 2, ...)
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")
        
        if device_id >= torch.cuda.device_count():
            raise ValueError(
                f"GPU {device_id} not available. "
                f"Only {torch.cuda.device_count()} GPU(s) found."
            )
        
        super().__init__(device_id)
    
    def _initialize_device(self) -> torch.device:
        """
        CUDA 디바이스 초기화
        """
        return torch.device(f"cuda:{self.device_id}")
    
    def _get_name(self) -> str:
        """
        GPU 이름 반환 (예: "CUDA:0 (NVIDIA RTX 3090)")
        """
        gpu_name = torch.cuda.get_device_name(self.device_id)
        return f"CUDA:{self.device_id} ({gpu_name})"
    
    def is_available(self) -> bool:
        """
        CUDA 사용 가능 여부
        """
        return torch.cuda.is_available() and self.device_id < torch.cuda.device_count()
    
    def get_memory_usage(self) -> float:
        """
        현재 GPU 메모리 사용량 반환
        
        Returns:
            float: 메모리 사용량 (MB)
        """
        return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
    
    def reset_peak_memory(self):
        """
        GPU 피크 메모리 통계 리셋
        """
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
    
    def _synchronize(self):
        """
        GPU 연산 완료 대기 (중요!)
        비동기 연산이 완료될 때까지 대기
        """
        torch.cuda.synchronize(self.device)
    
    def get_info(self) -> dict:
        """
        GPU 정보 반환
        
        Returns:
            dict: GPU 상세 정보
        """
        props = torch.cuda.get_device_properties(self.device_id)
        
        return {
            "device": f"CUDA:{self.device_id}",
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / (1024 ** 3),
            "multi_processor_count": props.multi_processor_count,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
        }
    
    def get_memory_info(self) -> dict:
        """
        현재 GPU 메모리 상태 반환
        
        Returns:
            dict: 메모리 정보 (allocated, reserved, free)
        """
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)
        free = total - allocated
        
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "free_mb": free,
            "total_mb": total,
            "utilization_percent": (allocated / total) * 100
        }
    
    def __str__(self):
        info = self.get_info()
        mem_info = self.get_memory_info()
        return (f"GPUAccelerator("
                f"device={info['name']}, "
                f"memory={mem_info['allocated_mb']:.0f}/{mem_info['total_mb']:.0f}MB)")


# 편의 함수들
def create_gpu_accelerator(device_id: int = 0) -> GPUAccelerator:
    """
    GPU Accelerator 생성 헬퍼 함수
    
    Args:
        device_id: GPU ID
    
    Returns:
        GPUAccelerator 인스턴스
    """
    return GPUAccelerator(device_id=device_id)


def get_available_gpus() -> list:
    """
    사용 가능한 GPU 목록 반환
    
    Returns:
        list: GPU ID 리스트
    """
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def create_all_gpu_accelerators() -> list:
    """
    모든 사용 가능한 GPU에 대해 accelerator 생성
    
    Returns:
        list: GPUAccelerator 인스턴스 리스트
    """
    return [GPUAccelerator(device_id=i) for i in get_available_gpus()]

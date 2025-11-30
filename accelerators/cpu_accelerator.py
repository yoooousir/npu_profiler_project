"""
CPU Accelerator Implementation
CPU를 사용한 모델 프로파일링
"""

import torch
import psutil
import os
from .base import BaseAccelerator


class CPUAccelerator(BaseAccelerator):
    """
    CPU Accelerator
    
    특징:
    - 항상 사용 가능
    - 메모리는 프로세스 메모리로 측정
    - 멀티스레딩 지원 (torch.set_num_threads)
    """
    
    def __init__(self, device_id: int = 0, num_threads: int = None):
        """
        Args:
            device_id: CPU는 항상 0 (호환성을 위해 유지)
            num_threads: PyTorch CPU 스레드 수 (None이면 자동)
        """
        self.num_threads = num_threads
        self.process = psutil.Process(os.getpid())
        self.initial_memory = 0  # 초기 메모리 기준점
        
        # 스레드 수 설정
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        
        super().__init__(device_id)
    
    def _initialize_device(self) -> torch.device:
        """
        CPU 디바이스 초기화
        """
        return torch.device("cpu")
    
    def _get_name(self) -> str:
        """
        Accelerator 이름 반환
        """
        threads = self.num_threads or torch.get_num_threads()
        return f"CPU (threads={threads})"
    
    def is_available(self) -> bool:
        """
        CPU는 항상 사용 가능
        """
        return True
    
    def get_memory_usage(self) -> float:
        """
        현재 프로세스의 메모리 사용량 반환
        
        Returns:
            float: 메모리 사용량 (MB)
        """
        # RSS (Resident Set Size) 사용
        memory_info = self.process.memory_info()
        current_memory = memory_info.rss / (1024 ** 2)  # Bytes to MB
        
        # 초기 메모리 대비 증가량 반환
        return current_memory - self.initial_memory
    
    def reset_peak_memory(self):
        """
        메모리 측정 기준점 리셋
        """
        memory_info = self.process.memory_info()
        self.initial_memory = memory_info.rss / (1024 ** 2)
    
    def _synchronize(self):
        """
        CPU는 동기화 불필요
        """
        pass
    
    def get_info(self) -> dict:
        """
        CPU 정보 반환
        
        Returns:
            dict: CPU 정보 (코어 수, 스레드 수 등)
        """
        return {
            "device": "CPU",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "torch_threads": torch.get_num_threads(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A",
            "total_memory_gb": psutil.virtual_memory().total / (1024 ** 3)
        }
    
    def __str__(self):
        info = self.get_info()
        return (f"CPUAccelerator("
                f"cores={info['physical_cores']}/{info['logical_cores']}, "
                f"threads={info['torch_threads']})")


# 편의 함수
def create_cpu_accelerator(num_threads: int = None) -> CPUAccelerator:
    """
    CPU Accelerator 생성 헬퍼 함수
    
    Args:
        num_threads: PyTorch CPU 스레드 수
    
    Returns:
        CPUAccelerator 인스턴스
    """
    return CPUAccelerator(num_threads=num_threads)

"""
NPU (Neural Processing Unit) Accelerator Implementation
실제 NPU 하드웨어 없이 CPU로 시뮬레이션
"""

import torch
import random
from .base import BaseAccelerator


class NPUAccelerator(BaseAccelerator):
    """
    NPU Accelerator (시뮬레이션)
    
    특징:
    - 실제 NPU 하드웨어 없이 CPU로 동작
    - 성능 특성을 시뮬레이션 (CPU보다 약간 느리게 설정)
    - FuriosaAI, Rebellions 같은 NPU 벤더 시뮬레이션 가능
    
    실무에서는:
    - FuriosaAI SDK를 사용하면 실제 NPU와 통신
    - Rebellions ATOM SDK 사용
    """
    
    def __init__(self, device_id: int = 0, vendor: str = "generic", performance_factor: float = 0.8):
        """
        Args:
            device_id: NPU ID (멀티 NPU 시뮬레이션용)
            vendor: NPU 벤더 ("furiosa", "rebellions", "generic")
            performance_factor: CPU 대비 성능 배율 (0.8 = CPU의 80% 속도)
        """
        self.vendor = vendor.lower()
        self.performance_factor = performance_factor
        self.simulated = True  # 시뮬레이션 모드 표시
        
        super().__init__(device_id)
        
        print(f"⚠️  NPU Simulation Mode: Using CPU to simulate {self.vendor.upper()} NPU")
    
    def _initialize_device(self) -> torch.device:
        """
        NPU 디바이스 초기화 (실제로는 CPU 사용)
        """
        # 실제 NPU SDK가 있다면 여기서 초기화
        # 예: furiosa_sdk.Device(device_id)
        return torch.device("cpu")
    
    def _get_name(self) -> str:
        """
        NPU 이름 반환
        """
        vendor_names = {
            "furiosa": "FuriosaAI Warboy",
            "rebellions": "Rebellions ATOM",
            "generic": "Generic NPU"
        }
        vendor_name = vendor_names.get(self.vendor, "Unknown NPU")
        return f"NPU:{self.device_id} ({vendor_name} - Simulated)"
    
    def is_available(self) -> bool:
        """
        시뮬레이션 모드에서는 항상 사용 가능
        실제 NPU가 있다면: furiosa_sdk.is_available() 등 사용
        """
        return True
    
    def get_memory_usage(self) -> float:
        """
        NPU 메모리 사용량 반환 (시뮬레이션)
        실제로는 NPU 메모리 API 사용
        """
        # 시뮬레이션: 랜덤하게 메모리 사용량 생성
        # 실제: furiosa_sdk.get_memory_usage() 등
        return random.uniform(50, 200)  # 50-200 MB
    
    def reset_peak_memory(self):
        """
        NPU 메모리 리셋 (시뮬레이션)
        """
        # 시뮬레이션: 아무것도 안 함
        # 실제: furiosa_sdk.reset_memory_stats()
        pass
    
    def _synchronize(self):
        """
        NPU 연산 완료 대기 (시뮬레이션)
        """
        # 시뮬레이션: 아무것도 안 함
        # 실제: furiosa_sdk.synchronize()
        pass
    
    def profile_model(self, model, dataloader, max_batches=10, warmup_batches=2):
        """
        NPU 프로파일링 (시뮬레이션 모드에서 성능 조정)
        
        시뮬레이션 방법:
        1. CPU로 실제 추론 수행
        2. 측정된 시간에 performance_factor 적용
        3. NPU 특성 반영 (낮은 분산, 안정적인 성능)
        """
        print(f"\n🔸 Running in NPU simulation mode")
        print(f"🔸 Performance factor: {self.performance_factor}x of CPU")
        print(f"🔸 Vendor: {self.vendor.upper()}")
        
        # 부모 클래스의 profile_model 호출
        result = super().profile_model(model, dataloader, max_batches, warmup_batches)
        
        # 시뮬레이션: 시간 조정
        # NPU는 일반적으로 CPU보다 느리지만 안정적
        result.avg_time_ms /= self.performance_factor
        result.min_time_ms /= self.performance_factor
        result.max_time_ms /= self.performance_factor
        result.std_time_ms *= 0.5  # NPU는 분산이 작음 (안정적)
        
        # 처리량 재계산
        result.throughput_samples_per_sec *= self.performance_factor
        
        # 배치 시간도 조정
        result.batch_times = [t / self.performance_factor for t in result.batch_times]
        
        print(f"\n✅ NPU Simulation completed")
        return result
    
    def get_info(self) -> dict:
        """
        NPU 정보 반환 (시뮬레이션)
        """
        vendor_specs = {
            "furiosa": {
                "memory_gb": 8,
                "tops": 128,  # INT8 TOPS
                "power_w": 40
            },
            "rebellions": {
                "memory_gb": 16,
                "tops": 200,
                "power_w": 50
            },
            "generic": {
                "memory_gb": 4,
                "tops": 64,
                "power_w": 30
            }
        }
        
        specs = vendor_specs.get(self.vendor, vendor_specs["generic"])
        
        return {
            "device": f"NPU:{self.device_id}",
            "vendor": self.vendor.upper(),
            "simulated": self.simulated,
            "memory_gb": specs["memory_gb"],
            "performance_tops": specs["tops"],
            "power_consumption_w": specs["power_w"],
            "performance_factor": self.performance_factor
        }
    
    def __str__(self):
        info = self.get_info()
        return (f"NPUAccelerator("
                f"vendor={info['vendor']}, "
                f"simulated={info['simulated']}, "
                f"performance={self.performance_factor}x)")


# 편의 함수들
def create_npu_accelerator(
    device_id: int = 0, 
    vendor: str = "generic",
    performance_factor: float = 0.8
) -> NPUAccelerator:
    """
    NPU Accelerator 생성 헬퍼 함수
    
    Args:
        device_id: NPU ID
        vendor: "furiosa", "rebellions", "generic"
        performance_factor: CPU 대비 성능 배율
    
    Returns:
        NPUAccelerator 인스턴스
    """
    return NPUAccelerator(device_id=device_id, vendor=vendor, performance_factor=performance_factor)


def create_furiosa_npu(device_id: int = 0) -> NPUAccelerator:
    """FuriosaAI Warboy NPU 시뮬레이션"""
    return NPUAccelerator(device_id=device_id, vendor="furiosa", performance_factor=1.2)


def create_rebellions_npu(device_id: int = 0) -> NPUAccelerator:
    """Rebellions ATOM NPU 시뮬레이션"""
    return NPUAccelerator(device_id=device_id, vendor="rebellions", performance_factor=1.5)

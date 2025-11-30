"""
Base Accelerator Class
추상 클래스로, 모든 accelerator의 공통 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import time


@dataclass
class ProfilingResult:
    """
    프로파일링 결과를 담는 데이터 클래스
    """
    model_name: str
    accelerator_type: str
    device_name: str
    
    # 성능 지표
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    
    # 메모리 지표
    avg_memory_mb: float
    peak_memory_mb: float
    
    # 처리량
    throughput_samples_per_sec: float
    
    # 원본 데이터
    batch_times: List[float]
    batch_memories: List[float]
    
    # 레이어별 시간 (SimpleCNN만 해당)
    layer_times: Optional[Dict[str, float]] = None
    
    def __str__(self):
        """결과를 보기 좋게 출력"""
        result = f"""
{'='*60}
Profiling Results
{'='*60}
Model:           {self.model_name}
Accelerator:     {self.accelerator_type}
Device:          {self.device_name}

Performance Metrics:
  Average Time:  {self.avg_time_ms:.2f} ms
  Min Time:      {self.min_time_ms:.2f} ms
  Max Time:      {self.max_time_ms:.2f} ms
  Std Dev:       {self.std_time_ms:.2f} ms

Memory Metrics:
  Average Memory: {self.avg_memory_mb:.2f} MB
  Peak Memory:    {self.peak_memory_mb:.2f} MB

Throughput:      {self.throughput_samples_per_sec:.2f} samples/sec
{'='*60}
"""
        
        if self.layer_times:
            result += "\nLayer-wise Timing:\n"
            for layer, t in self.layer_times.items():
                result += f"  {layer:10s}: {t:.2f} ms\n"
            result += f"{'='*60}\n"
        
        return result


class BaseAccelerator(ABC):
    """
    모든 Accelerator의 베이스 클래스
    
    CPU, GPU, NPU 등 모든 accelerator는 이 클래스를 상속받아 구현합니다.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Args:
            device_id: 디바이스 ID (GPU의 경우 cuda:0, cuda:1 등)
        """
        self.device_id = device_id
        self.device = self._initialize_device()
        self.name = self._get_name()
    
    @abstractmethod
    def _initialize_device(self) -> torch.device:
        """
        디바이스 초기화
        각 accelerator가 구체적으로 구현해야 함
        
        Returns:
            torch.device: 초기화된 PyTorch 디바이스
        """
        pass
    
    @abstractmethod
    def _get_name(self) -> str:
        """
        Accelerator 이름 반환
        
        Returns:
            str: Accelerator 이름 (예: "CPU", "CUDA", "NPU")
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Accelerator 사용 가능 여부 확인
        
        Returns:
            bool: 사용 가능하면 True
        """
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> float:
        """
        현재 메모리 사용량 반환
        
        Returns:
            float: 메모리 사용량 (MB)
        """
        pass
    
    @abstractmethod
    def reset_peak_memory(self):
        """
        피크 메모리 통계 리셋
        """
        pass
    
    def profile_model(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 10,
        warmup_batches: int = 2
    ) -> ProfilingResult:
        """
        모델 프로파일링 (공통 로직)
        
        Args:
            model: 프로파일링할 PyTorch 모델
            dataloader: 데이터로더
            max_batches: 측정할 최대 배치 수
            warmup_batches: 워밍업 배치 수 (측정에서 제외)
        
        Returns:
            ProfilingResult: 프로파일링 결과
        """
        if not self.is_available():
            raise RuntimeError(f"{self.name} is not available on this system")
        
        # 모델을 디바이스로 이동
        model = model.to(self.device)
        model.eval()
        
        # 메모리 리셋
        self.reset_peak_memory()
        
        batch_times = []
        batch_memories = []
        layer_times_accumulator = []
        
        print(f"\n{'='*60}")
        print(f"Profiling: {type(model).__name__} on {self.name}")
        print(f"Device: {self.device}")
        print(f"{'='*60}")
        
        # SimpleCNN 여부 확인
        is_custom_model = hasattr(model, "forward") and "SimpleCNN" in type(model).__name__
        
        total_batches = warmup_batches + max_batches
        
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= total_batches:
                break
            
            data, target = data.to(self.device), target.to(self.device)
            
            # 추론 시간 측정
            start_time = time.time()
            with torch.no_grad():
                if is_custom_model:
                    output, timings = model(data, profile=True)
                    if batch_idx >= warmup_batches:  # 워밍업 이후만 기록
                        layer_times_accumulator.append(timings)
                else:
                    output = model(data)
                    timings = None
            
            # 디바이스 동기화 (GPU의 경우 중요)
            self._synchronize()
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # ms
            
            # 워밍업 단계는 통계에서 제외
            if batch_idx < warmup_batches:
                print(f"[Warmup {batch_idx}] Time: {inference_time:.2f} ms")
                continue
            
            # 메모리 사용량 측정
            memory_usage = self.get_memory_usage()
            
            batch_times.append(inference_time)
            batch_memories.append(memory_usage)
            
            actual_batch_idx = batch_idx - warmup_batches
            print(f"[Batch {actual_batch_idx}] Total: {inference_time:.2f} ms, Mem: {memory_usage:.2f} MB")
            
            if timings is not None:
                for layer, t in timings.items():
                    print(f"  Layer {layer}: {t:.2f} ms")
        
        # 통계 계산
        avg_time = sum(batch_times) / len(batch_times)
        min_time = min(batch_times)
        max_time = max(batch_times)
        
        # 표준편차 계산
        variance = sum((t - avg_time) ** 2 for t in batch_times) / len(batch_times)
        std_time = variance ** 0.5
        
        avg_memory = sum(batch_memories) / len(batch_memories) if batch_memories else 0
        peak_memory = max(batch_memories) if batch_memories else 0
        
        throughput = (1000 / avg_time) * dataloader.batch_size  # samples/sec
        
        # 레이어별 평균 시간 계산
        layer_times_avg = None
        if layer_times_accumulator:
            layer_names = list(layer_times_accumulator[0].keys())
            layer_times_avg = {}
            for layer in layer_names:
                times = [batch[layer] for batch in layer_times_accumulator]
                layer_times_avg[layer] = sum(times) / len(times)
        
        result = ProfilingResult(
            model_name=type(model).__name__,
            accelerator_type=self.name,
            device_name=str(self.device),
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_time_ms=std_time,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            throughput_samples_per_sec=throughput,
            batch_times=batch_times,
            batch_memories=batch_memories,
            layer_times=layer_times_avg
        )
        
        print(result)
        return result
    
    def _synchronize(self):
        """
        디바이스 동기화 (기본 구현: 아무것도 안 함)
        GPU accelerator에서 오버라이드
        """
        pass
    
    def __str__(self):
        return f"{self.name} (device={self.device})"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(device_id={self.device_id})"

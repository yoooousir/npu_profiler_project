"""
NPU Profiler - Main Script
새로운 Accelerator 아키텍처 사용
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import get_model
from accelerators import (
    CPUAccelerator,
    GPUAccelerator,
    create_furiosa_npu,
    create_rebellions_npu,
    get_available_gpus
)
import argparse
import json
import os
from datetime import datetime


# =====================================
# 설정
# =====================================
BATCH_SIZE = 32
MAX_BATCHES = 10
WARMUP_BATCHES = 2

MODELS = ["simplecnn", "resnet18", "mobilenet_v2"]


# =====================================
# 데이터 로더 준비
# =====================================
def prepare_dataloaders():
    """
    모델별 DataLoader 준비
    
    Returns:
        dict: 모델 이름을 키로 하는 DataLoader 딕셔너리
    """
    # SimpleCNN용: 28x28 grayscale
    transform_simplecnn = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset_simplecnn = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True,
        transform=transform_simplecnn
    )
    train_loader_simplecnn = DataLoader(
        train_dataset_simplecnn, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    # ResNet/MobileNet용: 224x224 RGB
    transform_torchvision = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    train_dataset_torchvision = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True,
        transform=transform_torchvision
    )
    train_loader_torchvision = DataLoader(
        train_dataset_torchvision, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    return {
        "simplecnn": train_loader_simplecnn,
        "resnet18": train_loader_torchvision,
        "mobilenet_v2": train_loader_torchvision
    }


# =====================================
# Accelerator 준비
# =====================================
def prepare_accelerators(use_npu=True, npu_vendor="furiosa"):
    """
    사용 가능한 모든 accelerator 준비
    
    Args:
        use_npu: NPU 시뮬레이션 사용 여부
        npu_vendor: NPU 벤더 ("furiosa" 또는 "rebellions")
    
    Returns:
        list: Accelerator 인스턴스 리스트
    """
    accelerators = []
    
    # 1. CPU (항상 사용 가능)
    accelerators.append(CPUAccelerator())
    
    # 2. GPU (CUDA 사용 가능하면)
    gpu_ids = get_available_gpus()
    for gpu_id in gpu_ids:
        accelerators.append(GPUAccelerator(device_id=gpu_id))
    
    # 3. NPU (시뮬레이션)
    if use_npu:
        if npu_vendor == "furiosa":
            accelerators.append(create_furiosa_npu())
        elif npu_vendor == "rebellions":
            accelerators.append(create_rebellions_npu())
        else:
            print(f"⚠️  Unknown NPU vendor: {npu_vendor}, skipping NPU")
    
    return accelerators


# =====================================
# 프로파일링 실행
# =====================================
def run_profiling(
    models=None,
    accelerators=None,
    dataloaders=None,
    max_batches=MAX_BATCHES,
    warmup_batches=WARMUP_BATCHES,
    save_results=True
):
    """
    모든 모델 × 모든 accelerator 조합으로 프로파일링
    
    Args:
        models: 프로파일링할 모델 리스트
        accelerators: 사용할 accelerator 리스트
        dataloaders: DataLoader 딕셔너리
        max_batches: 측정할 배치 수
        warmup_batches: 워밍업 배치 수
        save_results: 결과 저장 여부
    
    Returns:
        list: ProfilingResult 리스트
    """
    if models is None:
        models = MODELS
    
    if dataloaders is None:
        dataloaders = prepare_dataloaders()
    
    if accelerators is None:
        accelerators = prepare_accelerators()
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"Starting Profiling")
    print(f"{'='*70}")
    print(f"Models: {', '.join(models)}")
    print(f"Accelerators: {len(accelerators)}")
    print(f"Batches: {max_batches} (+ {warmup_batches} warmup)")
    print(f"{'='*70}\n")
    
    total_runs = len(models) * len(accelerators)
    current_run = 0
    
    for model_name in models:
        for accelerator in accelerators:
            current_run += 1
            
            print(f"\n[{current_run}/{total_runs}] ", end="")
            
            try:
                # 모델 로드
                model = get_model(model_name)
                
                # 적절한 DataLoader 선택
                dataloader = dataloaders.get(model_name)
                
                # 프로파일링 실행
                result = accelerator.profile_model(
                    model=model,
                    dataloader=dataloader,
                    max_batches=max_batches,
                    warmup_batches=warmup_batches
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error profiling {model_name} on {accelerator.name}: {e}")
                continue
    
    # 결과 저장
    if save_results:
        save_profiling_results(results)
    
    # 요약 출력
    print_summary(results)
    
    return results


# =====================================
# 결과 저장
# =====================================
def save_profiling_results(results):
    """
    프로파일링 결과를 JSON으로 저장
    
    Args:
        results: ProfilingResult 리스트
    """
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/profiling_results_{timestamp}.json"
    
    # JSON 직렬화 가능한 형태로 변환
    results_dict = []
    for result in results:
        results_dict.append({
            "model_name": result.model_name,
            "accelerator_type": result.accelerator_type,
            "device_name": result.device_name,
            "avg_time_ms": result.avg_time_ms,
            "min_time_ms": result.min_time_ms,
            "max_time_ms": result.max_time_ms,
            "std_time_ms": result.std_time_ms,
            "avg_memory_mb": result.avg_memory_mb,
            "peak_memory_mb": result.peak_memory_mb,
            "throughput_samples_per_sec": result.throughput_samples_per_sec,
            "layer_times": result.layer_times
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {filename}")


# =====================================
# 요약 출력
# =====================================
def print_summary(results):
    """
    프로파일링 결과 요약 출력
    
    Args:
        results: ProfilingResult 리스트
    """
    print(f"\n{'='*70}")
    print(f"Profiling Summary")
    print(f"{'='*70}")
    
    # 테이블 헤더
    print(f"{'Model':<15} {'Accelerator':<25} {'Avg Time (ms)':<15} {'Throughput (samples/s)':<25}")
    print(f"{'-'*15} {'-'*25} {'-'*15} {'-'*25}")
    
    # 각 결과 출력
    for result in results:
        print(f"{result.model_name:<15} "
              f"{result.accelerator_type:<25} "
              f"{result.avg_time_ms:<15.2f} "
              f"{result.throughput_samples_per_sec:<25.2f}")
    
    print(f"{'='*70}\n")


# =====================================
# CLI 인터페이스
# =====================================
def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="NPU Profiler - Multi-model Multi-accelerator Profiling"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["simplecnn", "resnet18", "mobilenet_v2"],
        default=MODELS,
        help="Models to profile"
    )
    
    parser.add_argument(
        "--batches",
        type=int,
        default=MAX_BATCHES,
        help="Number of batches to profile"
    )
    
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_BATCHES,
        help="Number of warmup batches"
    )
    
    parser.add_argument(
        "--no-npu",
        action="store_true",
        help="Disable NPU simulation"
    )
    
    parser.add_argument(
        "--npu-vendor",
        choices=["furiosa", "rebellions"],
        default="furiosa",
        help="NPU vendor to simulate"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to JSON"
    )
    
    return parser.parse_args()


# =====================================
# 메인 함수
# =====================================
def main():
    """메인 실행 함수"""
    args = parse_args()
    
    # DataLoader 준비
    dataloaders = prepare_dataloaders()
    
    # Accelerator 준비
    accelerators = prepare_accelerators(
        use_npu=not args.no_npu,
        npu_vendor=args.npu_vendor
    )
    
    # 프로파일링 실행
    results = run_profiling(
        models=args.models,
        accelerators=accelerators,
        dataloaders=dataloaders,
        max_batches=args.batches,
        warmup_batches=args.warmup,
        save_results=not args.no_save
    )
    
    print(f"\n✅ Profiling completed! Total runs: {len(results)}")


if __name__ == "__main__":
    main()
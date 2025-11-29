import torch
from torch import cuda
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime

def profile_model(model, dataloader, device, max_batches=10, save_plot=True):
    """
    모델 프로파일링: 시간, 메모리, 레이어별 분석
    
    Args:
        model: PyTorch 모델
        dataloader: DataLoader
        device: torch.device
        max_batches: 측정할 배치 수
        save_plot: 결과 시각화 저장 여부
    """
    model.eval()
    batch_times = []
    batch_mems = []
    layer_times_all = []  # 레이어별 시간 누적

    if device.type == "cuda":
        cuda.reset_peak_memory_stats(device)

    # 모델이 SimpleCNN이면 profile=True 사용
    is_custom_model = hasattr(model, "forward") and "SimpleCNN" in type(model).__name__

    print(f"\n{'='*60}")
    print(f"Profiling: {type(model).__name__} on {device}")
    print(f"{'='*60}")

    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        data, target = data.to(device), target.to(device)

        start_time = time.time()
        with torch.no_grad():
            if is_custom_model:
                output, timings = model(data, profile=True)
                layer_times_all.append(timings)
            else:
                output = model(data)
                timings = None
        end_time = time.time()

        total_time = (end_time - start_time) * 1000
        batch_times.append(total_time)

        if device.type == "cuda":
            mem = cuda.max_memory_allocated(device) / (1024 ** 2)
            batch_mems.append(mem)
        else:
            batch_mems.append(0)

        print(f"[Batch {batch_idx}] Total: {total_time:.2f} ms, Mem: {batch_mems[-1]:.2f} MB")
        if timings is not None:
            for layer, t in timings.items():
                print(f"  Layer {layer}: {t:.2f} ms")

    # 통계 계산
    avg_time = sum(batch_times) / len(batch_times)
    avg_mem = sum(batch_mems) / len(batch_mems) if batch_mems else 0
    throughput = (1000 / avg_time) * dataloader.batch_size  # samples/sec

    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}")
    print(f"Average Time:      {avg_time:.2f} ms")
    print(f"Average Memory:    {avg_mem:.2f} MB")
    print(f"Throughput:        {throughput:.2f} samples/sec")
    print(f"{'='*60}\n")

    # 시각화
    if save_plot:
        plot_results(
            model_name=type(model).__name__,
            device_name=str(device),
            batch_times=batch_times,
            batch_mems=batch_mems,
            layer_times_all=layer_times_all
        )

    return {
        "avg_time_ms": avg_time,
        "avg_memory_mb": avg_mem,
        "throughput_samples_per_sec": throughput,
        "batch_times": batch_times,
        "batch_mems": batch_mems
    }


def plot_results(model_name, device_name, batch_times, batch_mems, layer_times_all):
    """
    프로파일링 결과 시각화
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 레이어별 시간이 있으면 3개 subplot, 없으면 2개
    has_layer_times = len(layer_times_all) > 0
    n_plots = 3 if has_layer_times else 2
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    
    if n_plots == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes
    
    # Plot 1: Batch Time
    ax1.plot(batch_times, marker='o', linewidth=2, markersize=6)
    ax1.set_xlabel('Batch Index', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title(f'Inference Time per Batch\n{model_name} on {device_name}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=sum(batch_times)/len(batch_times), color='r', linestyle='--', 
                label=f'Avg: {sum(batch_times)/len(batch_times):.2f} ms')
    ax1.legend()
    
    # Plot 2: Memory Usage
    if sum(batch_mems) > 0:
        ax2.plot(batch_mems, marker='s', color='orange', linewidth=2, markersize=6)
        ax2.set_ylabel('Memory (MB)', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'No GPU Memory\n(CPU Mode)', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    ax2.set_xlabel('Batch Index', fontsize=12)
    ax2.set_title(f'Memory Usage\n{model_name} on {device_name}', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Layer Times (SimpleCNN only)
    if has_layer_times:
        # 레이어별 평균 시간 계산
        layer_names = list(layer_times_all[0].keys())
        layer_avg_times = {}
        for layer in layer_names:
            times = [batch[layer] for batch in layer_times_all]
            layer_avg_times[layer] = sum(times) / len(times)
        
        bars = ax3.bar(layer_avg_times.keys(), layer_avg_times.values(), 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax3.set_ylabel('Average Time (ms)', fontsize=12)
        ax3.set_title(f'Layer-wise Profiling\n{model_name}', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 막대 위에 값 표시
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 저장
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/profile_{model_name}_{device_name}_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {filename}")
    plt.close()
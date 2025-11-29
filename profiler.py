import torch
from torch import cuda
import time
import matplotlib.pyplot as plt

def profile_model(model, dataloader, device, max_batches=10):
    model.eval()
    batch_times = []
    batch_mems = []

    if device.type == "cuda":
        cuda.reset_peak_memory_stats(device)

    # 모델이 SimpleCNN이면 profile=True 사용
    is_custom_model = hasattr(model, "forward") and "SimpleCNN" in type(model).__name__

    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        data, target = data.to(device), target.to(device)

        start_time = time.time()
        with torch.no_grad():
            if is_custom_model:
                output, timings = model(data, profile=True)
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

import time
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

def measure_inference_time(model, data_loader, device, num_images=100):
    model.eval()

    # Warm-up
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= 10:
                break
            images = images.to(device)
            _ = model(images)

    # Measure
    times = []
    total_images = 0

    with torch.no_grad():
        for images, _ in data_loader:
            if total_images >= num_images:
                break

            images = images.to(device)
            batch_size = images.size(0)

            start_time = time.time()
            _ = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_time = time.time() - start_time

            times.append(batch_time)
            total_images += batch_size

    avg_time_per_batch = np.mean(times)
    avg_time_per_image_ms = (avg_time_per_batch / data_loader.batch_size) * 1000
    throughput = data_loader.batch_size / avg_time_per_batch

    return {
        'Avg Time per Image (ms)': avg_time_per_image_ms,
        'Throughput (images/sec)': throughput,
        'Total Time (100 images)': sum(times) * 1000
    }

def compute_metrics(predictions, labels, classes):
    report = classification_report(labels, predictions, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, predictions)
    return report, cm

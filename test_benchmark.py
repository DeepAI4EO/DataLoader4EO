import numpy as np
import litdata as ld
from Dataset4EO.channel_wise import create_channel_chunks
from Dataset4EO.image_wise import create_full_image
import time
import psutil
from Dataset4EO.channel_wise import ChannelWiseStreamingDataset
from Dataset4EO.image_wise import FullImageStreamingDataset

def create_channel_wise_image(index):
    """
    Create a channel-wise chunked dataset from a 10-channel multispectral image.
    """
    image_array = np.random.rand(256, 256, 10).astype(np.float32)  # Simulate reflectance data [0, 1]
    return create_channel_chunks(image_array, metadata={"label": np.random.randint(10)})

def prepare_datasets():
    """
    Create and optimize datasets for benchmarking.
    """
    
    # Full Image Dataset
    ld.optimize(
        fn=create_full_image,
        inputs=list(range(100)),  # Generate 10 samples
        output_dir="full_image_optimized_dataset",
        num_workers=4,
        chunk_bytes="128MB"
    )
    print("Full image dataset created successfully.")
    
    # Channel-Wise Dataset
    ld.optimize(
        fn=create_channel_wise_image,
        inputs=list(range(100)),  # Generate 10 samples
        output_dir="channel_wise_optimized_dataset",
        num_workers=4,
        chunk_bytes="128MB",
    )
    print("Channel-wise dataset created successfully.")


def benchmark_dataset(dataset, description, batch_size=32, num_samples=100):
    """
    Benchmark a dataset's performance.

    Args:
        dataset: The dataset to benchmark.
        description (str): Description of the dataset.
        batch_size (int): Batch size for testing.
        num_samples (int): Number of samples to process.

    Returns:
        dict: Benchmark results including time and memory usage.
    """
    from torch.utils.data import DataLoader

    process = psutil.Process()
    start_time = time.time()
    memory_before = process.memory_info().rss / (1024 * 1024)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    total_samples = 0
    for batch in dataloader:
        total_samples += len(batch["image"])
        if total_samples >= num_samples:
            break

    memory_after = process.memory_info().rss / (1024 * 1024)
    elapsed_time = time.time() - start_time

    results = {
        "description": description,
        "total_samples": total_samples,
        "time_elapsed": elapsed_time,
        "memory_usage_mb": memory_after - memory_before
    }
    return results


def test_benchmark():
    channel_wise_dataset = ChannelWiseStreamingDataset("channel_wise_optimized_dataset", channels_to_select=[0, 1, 2])
    full_image_dataset = FullImageStreamingDataset("full_image_optimized_dataset", channels_to_select=[0, 1, 2])

    channel_wise_results = benchmark_dataset(channel_wise_dataset, "Channel-Wise")
    full_image_results = benchmark_dataset(full_image_dataset, "Full Image")

    assert channel_wise_results["total_samples"] > 0
    assert full_image_results["total_samples"] > 0
    print(channel_wise_results)
    print(full_image_results)


if __name__=="__main__":
    prepare_datasets()
    test_benchmark()
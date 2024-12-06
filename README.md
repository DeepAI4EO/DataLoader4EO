<div  align="center">    
 <img src="resources/datasets4eo.png" width = "400" height = "130" alt="segmentation" align=center />
</div>


![example workflow](https://github.com/github/docs/actions/workflows/main.yml/badge.svg)

[Homepage of the project](https://earthnets.nicepage.io/)


# Dataset4EO: A Python Library for Efficient Dataset Management in Earth Observation

**Dataset4EO** is a Python library designed to streamline the creation, storage, and benchmarking of Earth observation datasets. The library focuses on two primary methods of handling large multi-channel remote sensing data:

- **Channel-Wise Storage**: Stores each channel of a multi-channel image as an independent chunk, allowing for selective decoding of specific channels.
- **Full-Image Storage**: Stores entire multi-channel images in chunks and selects specific channels during the decoding phase.

## Key Features

- **Channel-Wise Dataset Support**: Efficient storage and selective decoding of individual image channels.
- **Full-Image Dataset Support**: Traditional storage and decoding of entire multi-channel images.
- **Performance Benchmarking**: Tools to compare storage efficiency, memory usage, and decoding speed between channel-wise and full-image approaches.
- **Integration with LitData**: Fully leverages LitData’s streaming capabilities for handling large datasets.

## Installation

```bash
pip install -e .
```

## Usage Example

Dataset4EO is ideal for multispectral or hyperspectral datasets with more than three channels. By chunking image data channel-wise and loading them channel-wise from the disk for streaming, it reduces memory footprint and accelerates training. This is especially useful when you need to randomly select a subset of multispectral or hyperspectral image data. The library extends the functionality of LitData.

### Code Example

```python
import numpy as np
import dataset4eo as eodata

def create_channel_wise_image(index):
    # Create a channel-wise chunked dataset from a 10-channel multispectral image.
    # Prepare data
    image_array = np.random.rand(256, 256, 13)  # Multi-channel image data
    segmentation_map = np.random.randint(0, 5, (256, 256), dtype='uint8')
    depth_map = np.random.rand(256, 256).astype('float16')

    data_sample = {
        "image": image_array,
        "segmentation_map": segmentation_map,
        "depth_map": depth_map,
        "sample_id": index
    }
    return data_sample

if __name__ == "__main__":
    import os
    if not os.path.exists("optimized_channel_dataset"):
        # Channel-Wise Dataset
        eodata.optimize(
            fn=create_channel_wise_image,
            inputs=list(range(1000)),  # Generate 1000 samples
            output_dir="optimized_channel_dataset",
            num_workers=4,
            chunk_bytes="128MB",
        )

    # Initialize the dataset
    dataset = eodata.StreamingDataset(
        input_dir="optimized_channel_dataset",  # Directory where chunks are stored
        num_channels=13,
        channels_to_select=[0, 1, 2, 8, 4, 5]  # Channels to load
    )

    # Retrieve a sample
    sample = dataset[100]
    print(sample.keys())
    print(sample["image"].shape)            # Output: (256, 256, 6)
    print(sample["segmentation_map"].shape) # Output: (256, 256)
    print(sample["depth_map"].shape)        # Output: (256, 256)
    print(sample["sample_id"])              # Metadata
```

## Todo List

- Reorganize more than 400 datasets in the remote sensing community in a task-oriented way.
- Support high-level repositories for specific tasks such as object detection, segmentation, and more.
- Provide easy-to-use data loaders for custom projects.

---

**Dataset4EO** simplifies the management of Earth observation datasets, offering robust performance and seamless integration with LitData to handle large-scale remote sensing data more efficiently.

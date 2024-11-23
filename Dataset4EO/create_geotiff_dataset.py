import os
import numpy as np
import litdata as ld
from io import BytesIO
import rasterio

def create_channel_chunks(index):
    """
    Create chunks with channel-wise storage for a multi-channel image.
    """
    # Simulate a multi-channel image (e.g., Sentinel-2 with 10 channels)
    image_array = np.random.rand(256, 256, 10).astype(np.float32)  # Reflectance data [0, 1]

    chunks = {}
    for channel_idx in range(image_array.shape[-1]):
        # Save each channel to an in-memory buffer
        buffer = BytesIO()
        with rasterio.open(
            buffer,
            "w",
            driver="GTiff",
            height=image_array.shape[0],
            width=image_array.shape[1],
            count=1,  # Single channel
            dtype="float32",
            compress="deflate"  # Lossless compression
        ) as dst:
            dst.write(image_array[:, :, channel_idx], 1)  # Write the single channel

        buffer.seek(0)
        chunks[f"channel_{channel_idx}"] = buffer.read()  # Store channel in chunks

    # Metadata
    chunks["label"] = np.random.randint(10)  # Example label
    return chunks

if __name__ == "__main__":
    ld.optimize(
        fn=create_channel_chunks,
        inputs=list(range(10)),  # Simulate 10 images
        output_dir="channel_wise_optimized_dataset",
        num_workers=4,
        chunk_bytes="128MB"
    )

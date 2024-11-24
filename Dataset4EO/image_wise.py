import litdata as ld
import rasterio
import numpy as np
from io import BytesIO

def create_full_image(index):
    """
    Create a full 10-channel multispectral GeoTIFF image and return it as tar bytes.
    """
    # Simulate a multispectral image (e.g., Sentinel-2 with 10 channels)
    image_array = np.random.rand(256, 256, 10).astype(np.float32)  # Reflectance data [0, 1]

    # Save the full image as a GeoTIFF to an in-memory buffer
    buffer = BytesIO()
    with rasterio.open(
        buffer,
        "w",
        driver="GTiff",
        height=image_array.shape[0],
        width=image_array.shape[1],
        count=image_array.shape[2],
        dtype="float32",
        compress="deflate"  # Lossless compression
    ) as dst:
        for i in range(image_array.shape[2]):
            dst.write(image_array[:, :, i], i + 1)  # Write each channel

    buffer.seek(0)
    return {"index": index, "data": buffer.read(), "label": np.random.randint(10)}

class FullImageStreamingDataset(ld.StreamingDataset):
    """
    Full-image streaming dataset for LitData.
    """
    def __init__(self, input_dir, channels_to_select, **kwargs):
        super().__init__(input_dir, **kwargs)
        self.channels_to_select = channels_to_select

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        full_image_bytes = sample["data"]

        with rasterio.open(BytesIO(full_image_bytes)) as src:
            channels = [src.read(channel + 1) for channel in self.channels_to_select]
            sample["image"] = np.stack(channels, axis=-1)
        return sample

import litdata as ld
import rasterio
import numpy as np
from io import BytesIO


# Should try the HDF5 for random index and compress, the current version is slower than full image
def create_channel_chunks(image_array, metadata=None):
    """
    Create channel-wise chunks from a multi-channel image.

    Args:
        image_array (numpy.ndarray): Multi-channel image array of shape (H, W, C).
        metadata (dict): Additional metadata to include with the chunks.

    Returns:
        dict: A dictionary where each channel is stored as a separate chunk.
    """
    chunks = {}
    for channel_idx in range(image_array.shape[-1]):
        buffer = BytesIO()
        with rasterio.open(
            buffer,
            "w",
            driver="GTiff",
            height=image_array.shape[0],
            width=image_array.shape[1],
            count=1,
            dtype=image_array.dtype,
            compress="deflate"
        ) as dst:
            dst.write(image_array[..., channel_idx], 1)
        buffer.seek(0)
        chunks[f"channel_{channel_idx}"] = buffer.read()

    if metadata:
        chunks.update(metadata)

    return chunks


class ChannelWiseStreamingDataset(ld.StreamingDataset):
    """
    Channel-wise streaming dataset extension for LitData.
    """
    def __init__(self, input_dir, channels_to_select, **kwargs):
        super().__init__(input_dir, **kwargs)
        self.channels_to_select = channels_to_select

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        channels = []

        for channel_idx in self.channels_to_select:
            channel_key = f"channel_{channel_idx}"
            channel_data = sample[channel_key]

            with rasterio.open(BytesIO(channel_data)) as src:
                channels.append(src.read(1))

        sample["image"] = np.stack(channels, axis=-1)
        return sample
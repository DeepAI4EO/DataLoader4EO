import litdata as ld
import rasterio
import numpy as np
from io import BytesIO

class ChannelWiseStreamingDataset(ld.StreamingDataset):
    """
    A LitData StreamingDataset for channel-wise image data with selective decoding.
    """
    def __init__(self, input_dir, channels_to_select, **kwargs):
        """
        Initialize the dataset.

        Args:
            input_dir (str): Path to the LitData optimized dataset.
            channels_to_select (list[int]): List of channel indices to decode.
            kwargs: Additional arguments for LitData's StreamingDataset.
        """
        super().__init__(input_dir, **kwargs)
        self.channels_to_select = channels_to_select

    def __getitem__(self, index):
        """
        Retrieve a sample, decoding only the selected channels.
        """
        sample = super().__getitem__(index)
        channels = []

        # Decode only the selected channels
        for channel_idx in self.channels_to_select:
            channel_key = f"channel_{channel_idx}"
            channel_data = sample[channel_key]  # Get the raw bytes for the channel

            # Decode GeoTIFF from the bytes
            with rasterio.open(BytesIO(channel_data)) as src:
                channels.append(src.read(1))  # Read the single channel

        # Stack the selected channels
        sample["image"] = np.stack(channels, axis=-1)  # Combine channels into a multi-channel array
        return sample

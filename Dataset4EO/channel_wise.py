import litdata as ld
import numpy as np
import h5py
from io import BytesIO

def create_channel_chunks(image_array, metadata=None):
    """
    Create channel-wise chunks from a multi-channel image using HDF5.

    Args:
        image_array (numpy.ndarray): Multi-channel image array of shape (H, W, C).
        metadata (dict): Additional metadata to include with the chunks.

    Returns:
        dict: A dictionary containing HDF5 data stored as bytes.
    """
    buffer = BytesIO()
    with h5py.File(buffer, 'w') as hdf5_file:
        for channel_idx in range(image_array.shape[-1]):
            hdf5_file.create_dataset(
                f'channel_{channel_idx}',
                data=image_array[..., channel_idx],
                compression="gzip"
            )
    buffer.seek(0)
    chunks = {'hdf5_data': buffer.read()}

    if metadata:
        chunks.update(metadata)

    return chunks


class ChannelWiseStreamingDataset(ld.StreamingDataset):
    """
    Channel-wise streaming dataset extension using HDF5 for efficient channel access.
    """
    def __init__(self, input_dir, channels_to_select, **kwargs):
        super().__init__(input_dir, **kwargs)
        self.channels_to_select = channels_to_select

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        hdf5_data = sample['hdf5_data']
        buffer = BytesIO(hdf5_data)
        channels = []

        with h5py.File(buffer, 'r') as hdf5_file:
            for channel_idx in self.channels_to_select:
                channel_data = hdf5_file[f'channel_{channel_idx}'][:]
                channels.append(channel_data)

        sample["image"] = np.stack(channels, axis=-1)
        return sample

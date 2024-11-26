# my_pytorch_package/utils/hdf5_utils.py

import h5py
import numpy as np
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
import pdb

def create_hdf5_chunk(
    image_array: np.ndarray,
    other_data: Dict[str, np.ndarray],
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a data chunk with image data stored channel-wise and other data stored normally.

    Args:
        image_array (np.ndarray): Multi-channel image array of shape (H, W, C).
        other_data (Dict[str, np.ndarray]): Dictionary of other data arrays (e.g., segmentation maps).
        compression (Optional[str]): Compression algorithm. Defaults to 'gzip'.
        compression_opts (Optional[int]): Compression level. Defaults to None.
        metadata (Optional[Dict[str, Any]]): Additional metadata.

    Returns:
        Dict[str, Any]: A chunk containing HDF5 data and metadata.
    """
    buffer = BytesIO()
    with h5py.File(buffer, 'w') as hdf5_file:
        # Store image channels separately
        for channel_idx in range(image_array.shape[-1]):
            hdf5_file.create_dataset(
                f'channel_{channel_idx}',
                data=image_array[..., channel_idx],
                compression=compression,
                compression_opts=compression_opts
            )
        '''
        # Store other data normally
        for key, data in other_data.items():
            hdf5_file.create_dataset(
                name=key,
                data=data,
                compression=compression,
                compression_opts=compression_opts,
                dtype=data.dtype
            )
        '''
    buffer.seek(0)
    chunk = {'hdf5_data': buffer.read()}
    if metadata:
        chunk.update(metadata)
    if other_data:
        chunk.update(other_data)
    return chunk

def load_hdf5_chunk(
    hdf5_bytes: bytes,
    channels_to_select: Optional[List[int]] = None,
    other_keys: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Load selected channels and other datasets from HDF5 bytes.

    Args:
        hdf5_bytes (bytes): The HDF5 data as bytes.
        channels_to_select (Optional[List[int]]): List of channel indices to load.
        other_keys (Optional[List[str]]): List of other dataset names to load.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing loaded data.
    """
    buffer = BytesIO(hdf5_bytes)
    data_dict = {}
    with h5py.File(buffer, 'r') as hdf5_file:
        # Load selected channels
        if channels_to_select is not None:
            channels = []
            for channel_idx in channels_to_select:
                channel_name = f'channel_{channel_idx}'
                if channel_name in hdf5_file:
                    channels.append(hdf5_file[channel_name][:])
                else:
                    raise KeyError(f"Channel '{channel_name}' not found in HDF5 file.")
            data_dict['image'] = np.stack(channels, axis=-1)
        '''
        # Load other datasets
        if other_keys is not None:
            for key in other_keys:
                if key in hdf5_file:
                    data_dict[key] = hdf5_file[key][:]
                else:
                    raise KeyError(f"Dataset '{key}' not found in HDF5 file.")
        '''
    return data_dict

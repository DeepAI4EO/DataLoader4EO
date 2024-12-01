
#multichannel.py
import litdata as ld # type: ignore
from typing import Any, Callable, Dict, List, Optional
from .utils import load_hdf5_chunk

class ChannelwiseDataset(ld.StreamingDataset):
    """
    General-purpose HDF5 streaming dataset for PyTorch.

    This dataset allows efficient random access to channel-wise stored image data and other datasets.
    """

    def __init__(
        self,
        input_dir: str,
        channels_to_select: List[int],
        **kwargs
    ):
        """
        Initialize the ChannelwiseDataset.

        Args:
            input_dir (str): Directory containing the dataset chunks.
            channels_to_select (List[int]): List of channel indices to load.
            transform (Optional[Callable]): Function to transform the loaded data.
            **kwargs: Additional arguments for StreamingDataset.
        """
        super().__init__(input_dir, **kwargs)
        self.channels_to_select = channels_to_select

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)
        hdf5_data = sample.get('hdf5_data')

        if not hdf5_data:
            raise ValueError("Missing 'hdf5_data' in sample.")

        data_dict = load_hdf5_chunk(
            hdf5_data,
            channels_to_select=self.channels_to_select,
        )
        del sample["hdf5_data"]

        sample.update(data_dict)
        return sample

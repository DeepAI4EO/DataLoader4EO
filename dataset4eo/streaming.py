from typing import Any, Dict, List, Optional
from dataset4eo.multichannel import ChannelwiseDataset
import litdata as ld #type: ignore

class StreamingDataset:
    """
    Adaptive dataset class for handling both single-channel and multi-channel image datasets.

    This class decides whether to use the base `litdata.StreamingDataset` for up to 3 channels
    or the custom `ChannelwiseDataset` for more than 3 channels.
    """

    def __new__(
        cls,    
        input_dir: str,
        num_channels: int,
        channels_to_select: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Return the appropriate dataset instance based on the number of channels.

        Args:
            input_dir (str): Path to the dataset directory.
            num_channels (int): Number of channels in the image data.
            channels_to_select (Optional[List[int]]): Channel indices to load (for multi-channel data).
            **kwargs: Additional arguments for `ld.StreamingDataset` or `ChannelwiseDataset`.

        Returns:
            An instance of `ld.StreamingDataset` or `ChannelwiseDataset`.
        """
        num_channels = num_channels
        _is_multichannel = num_channels > 3

        if _is_multichannel:
            if channels_to_select is None:
                raise ValueError("`channels_to_select` must be provided for multi-channel data.")
            # Use the custom `ChannelwiseDataset`
            return ChannelwiseDataset(
                input_dir=input_dir,
                channels_to_select=channels_to_select,
                **kwargs
            )
        else:
            if channels_to_select is not None:
                raise ValueError("`channels_to_select` is not supported for #channles <= 3.")
            # Return an instance of ld.StreamingDataset for up to 3 channels
            return ld.StreamingDataset(
                input_dir=input_dir,
                **kwargs
            )
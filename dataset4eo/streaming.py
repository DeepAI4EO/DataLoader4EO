from typing import Any, Dict, List, Optional
import litdata as ld
from dataset4eo.multichannel import ChannelwiseDataset


class StreamingDataset(ld.StreamingDataset):
    """
    Adaptive dataset class for handling both single-channel and multi-channel image datasets.

    This class decides whether to use the base `litdata.StreamingDataset` for up to 3 channels
    or the custom `ChannelwiseDataset` for more than 3 channels.
    """

    def __init__(
        self,
        input_dir: str,
        num_channels: int,
        channels_to_select: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Initialize the StreamingDataset.

        Args:
            input_dir (str): Path to the dataset directory.
            num_channels (int): Number of channels in the image data.
            channels_to_select (Optional[List[int]]): Channel indices to load (for multi-channel data).
            **kwargs: Additional arguments for `litdata.StreamingDataset` or `ChannelwiseDataset`.
        """
        self.num_channels = num_channels
        self._is_multichannel = num_channels > 3

        if self._is_multichannel:
            if channels_to_select is None:
                raise ValueError("`channels_to_select` must be provided for multi-channel data.")
            self.channels_to_select = channels_to_select
            # Use the custom `ChannelwiseDataset`
            self._channelwise_dataset = ChannelwiseDataset(
                input_dir=input_dir,
                channels_to_select=channels_to_select,
                **kwargs
            )
        else:
            if channels_to_select is not None:
                raise ValueError("`channels_to_select` is not supported for #channles <= 3.")
            # Directly initialize the base `litdata.StreamingDataset`
            super().__init__(input_dir=input_dir, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieve an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Dict[str, Any]: The dataset sample.
        """
        if self._is_multichannel:
            return self._channelwise_dataset[index]
        return super().__getitem__(index)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        if self._is_multichannel:
            return len(self._channelwise_dataset)
        return super().__len__()

from GeoTiffStreaming import ChannelWiseStreamingDataset
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")


def test_channel_wise_streaming_dataset():
    """
    Test the ChannelWiseStreamingDataset for channel-wise decoding.
    """
    input_dir = "channel_wise_optimized_dataset"  # Path to the optimized dataset
    channels_to_select = [0, 1, 2, 6, 8]  # Select specific channels
    batch_size = 2

    # Initialize dataset and dataloader
    dataset = ChannelWiseStreamingDataset(input_dir=input_dir, channels_to_select=channels_to_select)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)  # Single-threaded for testing

    for batch in dataloader:
        # Extract images and labels
        images = batch["image"]
        labels = batch["label"]

        # Validate shapes and data consistency
        assert len(images) == batch_size, f"Expected batch size {batch_size}, but got {len(images)}"
        assert images.shape[-1] == len(channels_to_select), f"Expected {len(channels_to_select)} channels, but got {images.shape[-1]}"
        assert labels.shape[0] == batch_size, "Mismatch in label batch size"

        print(f"Image shape: {images[0].shape}, Label: {labels[0]}")
        #break

if __name__ == "__main__":
    test_channel_wise_streaming_dataset()

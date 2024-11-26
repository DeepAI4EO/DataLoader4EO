from utils import create_hdf5_chunk
from multichannel import ChannelwiseDataset
import numpy as np
import litdata as ld

def create_channel_wise_image(index):
    """
    Create a channel-wise chunked dataset from a 10-channel multispectral image.
    """
    # Prepare data
    image_array = np.random.rand(256, 256, 10)  # Multi-channel image data
    segmentation_map = np.random.randint(0, 5, (256, 256), dtype='uint8')
    depth_map = np.random.rand(256, 256).astype('float16')

    metadata = {'sample_id': index}

    # Create other data dictionary
    other_data = {
        'segmentation_map': segmentation_map,
        'depth_map': depth_map
    }

    chunk = create_hdf5_chunk(
        image_array=image_array,
        other_data=other_data,
        compression='gzip',
        metadata=metadata
    )
    return chunk


if __name__=="__main__":
    import os
    if not os.path.exists("channel_wise_optimized_dataset"):
    # Channel-Wise Dataset
        ld.optimize(
            fn=create_channel_wise_image,
            inputs=list(range(1000)),  # Generate 10 samples
            output_dir="channel_wise_optimized_dataset",
            num_workers=4,
            chunk_bytes="128MB",
        )

    # Save chunk to disk or integrate with LitData as needed
    # For example, save to a JSONL file or database where each line is a chunk

    # Initialize the dataset
    dataset = ChannelwiseDataset(
        input_dir='channel_wise_optimized_dataset',  # Directory where chunks are stored
        channels_to_select=[0, 1, 2, 8, 4, 5],  # Channels to load
        other_keys=['segmentation_map', 'depth_map'],  # Other datasets to load
        transform=None  # Optional transform
    )

    # Retrieve a sample
    sample = dataset[100]
    print(sample['image'].shape)  # Output: (256, 256, 3)
    print(sample['segmentation_map'].shape)  # Output: (256, 256)
    print(sample['depth_map'].shape)  # Output: (256, 256)
    print(sample['sample_id'])  # Metadata

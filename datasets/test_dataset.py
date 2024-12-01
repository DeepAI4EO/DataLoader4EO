import numpy as np
import dataset4eo as eodata

def create_channel_wise_image(index):
    """
    Create a channel-wise chunked dataset from a 10-channel multispectral image.
    """
    # Prepare data
    image_array = np.random.rand(256, 256, 13)  # Multi-channel image data
    segmentation_map = np.random.randint(0, 5, (256, 256), dtype='uint8')
    depth_map = np.random.rand(256, 256).astype('float16')

    data_sample = {"image":image_array,
                   'segmentation_map': segmentation_map,
                   'depth_map': depth_map,
                   'sample_id': index}

    return data_sample


if __name__=="__main__":
    import os
    if not os.path.exists("optimized_channel_dataset"):
    # Channel-Wise Dataset
        eodata.optimize(
            fn=create_channel_wise_image,
            inputs=list(range(1000)),  # Generate 10 samples
            output_dir="optimized_channel_dataset",
            num_workers=4,
            chunk_bytes="128MB",
        )

    #'''
    # Save chunk to disk or integrate with LitData as needed
    # For example, save to a JSONL file or database where each line is a chunk

    # Initialize the dataset
    dataset = eodata.StreamingDataset(
        input_dir='optimized_channel_dataset',  # Directory where chunks are stored
        num_channels = 13,
        channels_to_select=[0, 1, 2, 8, 4, 5],  # Channels to load
    )

    # Retrieve a sample
    sample = dataset[100]
    print(sample.keys())
    print(sample['image'].shape)  # Output: (256, 256, 3)
    print(sample['segmentation_map'].shape)  # Output: (256, 256)
    print(sample['depth_map'].shape)  # Output: (256, 256)
    print(sample['sample_id'])  # Metadata
    #'''
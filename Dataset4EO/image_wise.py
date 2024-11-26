import litdata as ld
import rasterio
import numpy as np
from io import BytesIO

def create_full_image(index):
    """
    Create a full 10-channel multispectral GeoTIFF image and return it as tar bytes.
    """
    # Simulate a multispectral image (e.g., Sentinel-2 with 10 channels)
    image_array = np.random.rand(256, 256, 200).astype(np.uint8)  # Reflectance data [0, 1]
    return {"index": index, "data": image_array, "label": np.random.randint(10)}

class FullImageStreamingDataset(ld.StreamingDataset):
    """
    Full-image streaming dataset for LitData.
    """
    def __init__(self, input_dir, channels_to_select, **kwargs):
        super().__init__(input_dir, **kwargs)
        self.channels_to_select = channels_to_select

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        full_image = sample["data"]
        sample["image"] = full_image[:,:,self.channels_to_select]
        return sample

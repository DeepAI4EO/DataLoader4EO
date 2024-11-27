import os
import h5py
from torch.utils.data import Dataset
import json
import numpy as np
import litdata as ld
from utils import create_hdf5_chunk


class GamusDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root_dir, 'images', split)
        self.class_dir = os.path.join(root_dir, 'classes', split)
        self.height_dir = os.path.join(root_dir, 'heights', split)

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.h5')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        base_name = img_file[:-6]  # Remove 'IMG.h5'

        # Load image
        img_path = os.path.join(self.image_dir, img_file)
        with h5py.File(img_path, 'r') as f:
            image = f['image'][()]  # Assuming the image is stored as a numpy array

        # Load class
        cls_file = f"{base_name}CLS.h5"
        cls_path = os.path.join(self.class_dir, cls_file)
        with h5py.File(cls_path, 'r') as f:
            class_label = f['image'][()]

        # Load height
        agl_file = f"{base_name}AGL.h5"
        agl_path = os.path.join(self.height_dir, agl_file)
        with h5py.File(agl_path, 'r') as f:
            height = f['image'][()]

        return (base_name, image, class_label, height)



############################-Create optimized Dataset-###############################

root_dir = '/home/xshadow/Datasets/GAMUS'  # Path to the original GAMUS dataset
output_dir = './optimized_gamus_dataset2'

split = "val"

dataset = GamusDataset(root_dir=root_dir, split=split)

def create_channel_wise_image(index):
    """
    Create a channel-wise chunked dataset from a 10-channel multispectral image.
    """
    # Prepare data
    basename, image, class_label, height = dataset[index]
    # Convert the image to JPEG
    image_array = image.astype(np.uint8)  # Convert to HWC
    # Convert class label to bytes
    class_data = class_label.astype(np.uint8)
    # Convert height map to bytes
    height_data = height.astype(np.float16)

    metadata = {'sample_id': index, "base_name": basename, "split": "train", "geolocation": (-1,-1)}

    # Create other data dictionary
    other_data = {
        'segmentation_map': class_data,
        'height_map': height_data
    }

    chunk = create_hdf5_chunk(
        image_array=image_array,
        other_data=other_data,
        compression='gzip',
        metadata=metadata
    )
    return chunk

if __name__=="__main__":
    metadata = {
        "Dataset": "GAMUS",
        "split": split,
        "num_samples": len(dataset),
        "attributes": {
            "image": {"dtype": "uint8", "format": "pil"},
            "class": {"dtype": "uint8", "format": "pil"},
            "height": {"dtype": "float16", "format": "numpy"},
        },
    }
    ld.optimize(
        fn=create_channel_wise_image,
        inputs=list(range(len(dataset))),
        output_dir=output_dir,
        num_workers=8,
        chunk_bytes="256MB",
    )
    # Save the metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(f"{split.capitalize()} dataset optimized and metadata saved to {metadata_path}")
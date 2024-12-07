import os
import h5py # type: ignore
from torch.utils.data import Dataset
import json
import numpy as np
import dataset4eo as eodata

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

split = "train"
root_dir = '/home/xshadow/Datasets/GAMUS'  # Path to the original GAMUS dataset
output_dir = f'./gamus_dataset_{split}'


dataset = GamusDataset(root_dir=root_dir, split=split)

def process_sample(index):
    """
    Process a single sample from the GAMUS dataset and return LitData-compatible data.
    """
    basename, image, class_label, height = dataset[index]
    # Convert the image to JPEG
    image = image.astype(np.uint8)  # Convert to HWC
    assert image.shape[-1] == 3
    # Convert class label to bytes
    class_data = class_label.astype(np.uint8)
    # Convert height map to bytes
    height_data = height.astype(np.float16)

    return {
        "name": basename,
        "image": image,
        "class": class_data,
        "height": height_data,
    }


metadata = {
    "Dataset": "GAMUS",
    "split": split,
    "num_samples": len(dataset),
    "attributes": {
        "name": {"dtype": "str"},
        "image": {"dtype": "uint8", "format": "pil"},
        "class": {"dtype": "uint8", "format": "pil"},
        "height": {"dtype": "float16", "format": "numpy"},
    },
}


if __name__=="__main__":
    eodata.optimize(
        fn=process_sample,
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

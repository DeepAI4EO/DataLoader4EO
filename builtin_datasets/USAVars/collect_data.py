import numpy as np
from torchgeo.datasets import USAVars as DataClass
import dataset4eo as eodata
import json
import os
import torch
from glob import glob
from skimage import io

# class DataClass(torch.utils.data.Dataset):
#     def __init__(self, root, split):
#         super(DataClass, self).__init__()
#         self.root = root
#         self.split = split

#         self.image_path_list = glob(os.path.join(root, "images", "*", "*.jpg"))

#         self.class_map = {v:k for k, v in CLASSES.items()}

#     def __getitem__(self, index):
#         image_path = self.image_path_list[index]
#         label = image_path.split("/")[-2]
#         image_name = image_path.split("/")[-1]
#         label_id = self.class_map[label]

#         image = io.imread(image_path)
#         return {
#             "name": image_name, 
#             "image": image, 
#             "label": np.array(label_id)
#         }

#     def __len__(self):
#         return len(self.image_path_list)

# Define the process function at the top level
def process_sample_fn(dataset, index):
    """
    Process a single sample from the So2Sat dataset and return LitData-compatible data.
    """
    sample = dataset[index]
    # Convert the image to JPEG

    ##########Change this accoring to the dims of images.
    image = sample["image"].numpy().transpose(1, 2, 0).astype(np.uint8)  # Convert to HWC 
    # image = sample["image"].astype(np.uint8)  # Convert to HWC 
    # Convert class label to bytes
    class_data = sample["labels"].numpy().astype(np.float32)

    return {
        "image": image,
        "treecover": class_data[0],
        "elevation": class_data[1],
        "population": class_data[2]
    }

# Create a wrapper for picklability
class ProcessSampleWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, index):
        return process_sample_fn(self.dataset, index)

if __name__ == "__main__":
    ########## Change the paths!
    root = "/data/chen/datasets/USAVars/"
    output_folder = "/data/chen/datasets/outputs/USAVars"

    os.makedirs(root, exist_ok=True)

    for split, split_name in zip(["train", "validation", "test"], ["train", "val", "test"]):
        output_dir = os.path.join(output_folder, split)
        os.makedirs(output_dir, exist_ok=True)

        output_flag = os.path.join(output_folder, split+".finish")
        if os.path.exists(output_flag):
            continue

        dataset = DataClass(root=root, split=split_name, download=True)

        ########## Change Metadata!!!
        metadata = {
            "Dataset": "USAVars",
            "split": split,
            "num_samples": len(dataset),
            "attributes": {
                "image": {"dtype": "uint8", "format": "numpy"},
                "treecover": {"dtype": "float32", "format": "numpy", "unit": "percentage"},
                "elevation": {"dtype": "float32", "format": "numpy", "unit": "meters"},
                "population": {"dtype": "float32", "format": "numpy", "unit": "people_per_square_kilometer"}
            },
                "bands": {
                    0: "Red",
                    1: "Green",
                    2: "Blue",
                    3: "NearInfrared",
                }
        }

        # Wrap the dataset in a picklable callable
        process_fn = ProcessSampleWrapper(dataset)

        # Optimize the dataset
        eodata.optimize(
            fn=process_fn,
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

        with open(output_flag, "w") as f:
            f.writelines("finished! \n")

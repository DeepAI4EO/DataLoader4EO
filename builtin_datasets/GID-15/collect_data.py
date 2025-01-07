import numpy as np
from torchgeo.datasets import GID15 as DataClass
import dataset4eo as eodata
import json
import os
import torch
from glob import glob
from skimage import io

CLASSES = {
    0: "background",
    1: "industrial_land",
    2: "urban_residential",
    3: "rural_residential",
    4: "traffic_land",
    5: "paddy_field",
    6: "irrigated_land",
    7: "dry_cropland",
    8: "garden_plot",
    9: "arbor_woodland",
    10: "shrub_land",
    11: "natural_grassland",
    12: "artificial_grassland",
    13: "river",
    14: "lake",
    15: "pond"
}

# class DataClass(torch.utils.data.Dataset):
#     def __init__(self, root, split):
#         super(DataClass, self).__init__()
#         self.root = root
#         self.split = split

#         self.image_path_list = glob(os.path.join(root, split, "*", "*.png"))

#         self.class_map = {
#             "High": 0,
#             "Low": 1,
#             "Moderate": 2,
#             "Non-burnable": 3,
#             "Very_High": 4,
#             "Very_Low": 5,
#             "Water": 6
#         }
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

    out = {"image": image}

    if "mask" in sample:
        class_data = sample["mask"].numpy().astype(np.uint8)
        out.update({
            "class": class_data
        })

    return out

# Create a wrapper for picklability
class ProcessSampleWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, index):
        return process_sample_fn(self.dataset, index)

if __name__ == "__main__":
    ########## Change the paths!
    root = "/data/chen/datasets/GID-15/"
    output_folder = "/data/chen/datasets/outputs/GID-15"

    for split, split_name in zip(["train", "validation", "test"], ["train", "val", "test"]):
        output_dir = os.path.join(output_folder, split)
        os.makedirs(output_dir, exist_ok=True)

        output_flag = os.path.join(output_folder, split+".finish")
        if os.path.exists(output_flag):
            continue

        dataset = DataClass(root=root, split=split_name)
        ########## Change Metadata!!!
        metadata = {
            "Dataset": "GID-15",
            "split": split,
            "num_samples": len(dataset),
            "attributes": {
                "image": {"dtype": "uint8", "format": "numpy"},
                "class": {"dtype": "uint8", "format": "numpy"},
                "classes": CLASSES,
            },
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

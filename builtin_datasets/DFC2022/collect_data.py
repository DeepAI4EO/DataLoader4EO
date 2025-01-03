import numpy as np
from torchgeo.datasets import DFC2022 as DataClass
import dataset4eo as eodata
import json
import os
import torch
from glob import glob
from skimage import io

CLASSES = {
    0: "No information",
    1: "Urban fabric",
    2: "Industrial, commercial, public, military, private and transport units",
    3: "Mine, dump and construction sites",
    4: "Artificial non-agricultural vegetated areas",
    5: "Arable land (annual crops)",
    6: "Permanent crops",
    7: "Pastures",
    8: "Complex and mixed cultivation patterns",
    9: "Orchards at the fringe of urban classes",
    10: "Forests",
    11: "Herbaceous vegetation associations",
    12: "Open spaces with little or no vegetation",
    13: "Wetlands",
    14: "Water",
    15: "Clouds and Shadows"
    }
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
    image = sample["image"].numpy().transpose(1, 2, 0).astype(np.uint8)[:, :, :3]  # Convert to HWC 
    height = sample["image"].numpy().transpose(1, 2, 0).astype(np.float16)[:, :, -1]
    # image = sample["image"].astype(np.uint8)  # Convert to HWC 
    # Convert class label to bytes

    out = {
        "image": image,
        "height": height,
    }
    if "mask" in sample:
        class_data = sample["mask"].numpy().astype(np.uint8)
        out.update({
            "class": class_data,
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
    root = "/data/chen/datasets/DFC2022/"
    output_folder = "/data/chen/datasets/outputs/DFC2022"

    os.makedirs(root, exist_ok=True)

    for split, split_name in zip(["train", "validation"], ["train", "val"]):
        output_dir = os.path.join(output_folder, split)
        os.makedirs(output_dir, exist_ok=True)

        output_flag = os.path.join(output_folder, split+".finish")
        if os.path.exists(output_flag):
            continue

        dataset = DataClass(root=root, split=split_name)
        ########## Change Metadata!!!
        metadata = {
            "Dataset": "DFC2022",
            "split": split,
            "num_samples": len(dataset),
            "attributes": {
                "image": {"dtype": "uint8", "format": "numpy"},
                "class": {"dtype": "uint8", "format": "numpy"},
                "height": {"dtype": "float16", "format": "numpy", "unit": "meters"},
                "classes": CLASSES
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

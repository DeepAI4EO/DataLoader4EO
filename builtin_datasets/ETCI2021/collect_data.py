import numpy as np
from torchgeo.datasets import ETCI2021 as DataClass
import dataset4eo as eodata
import json
import os
import torch
from glob import glob
from skimage import io

CLASSESWATER = {
    0: "no water body",
    1: "water body"
    }

CLASSESFLOOD = {
    0: "no flood",
    1: "flood"
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
    image = sample["image"].numpy().transpose(1, 2, 0).astype(np.uint8)  # Convert to HWC 
    # image = sample["image"].astype(np.uint8)  # Convert to HWC 
    # Convert class label to bytes

    out = {
        "image_vv": image[:, :, :3],
        "image_vh": image[:, :, 3:],

    }

    if "mask" in sample:
        mask = sample["mask"].numpy().transpose(1, 2, 0).astype(np.uint8)
        out.update({
            "mask_waterbody": mask[:, :, 0],
        })
        if mask.shape[-1] == 2:
            out.update({
                "mask_flood": mask[:, :, 1]
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
    root = "/data/chen/datasets/ETCI2021/"
    output_folder = "/data/chen/datasets/outputs/ETCI2021"

    os.makedirs(root, exist_ok=True)

    for split, split_name in zip(["train", "test"], ["train", "test"]):
        output_dir = os.path.join(output_folder, split)
        os.makedirs(output_dir, exist_ok=True)

        output_flag = os.path.join(output_folder, split+".finish")
        if os.path.exists(output_flag):
            continue

        dataset = DataClass(root=root, split=split_name, download=False)

        ########## Change Metadata!!!
        metadata = {
            "Dataset": "ETCI2021",
            "split": split,
            "num_samples": len(dataset),
            "attributes": {
                "image_vv": {"dtype": "uint8", "format": "numpy"},
                "image_vh": {"dtype": "uint8", "format": "numpy"},
                "mask_waterbody": {"dtype": "uint8", "format": "numpy"},
                "mask_flood": {"dtype": "uint8", "format": "numpy"},
                "waterbody": CLASSESWATER,
                "flood": CLASSESFLOOD
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

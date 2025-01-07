import numpy as np
# from torchgeo.datasets import GeoNRW as DataClass
import dataset4eo as eodata
import json
import os
import torch
from glob import glob
from skimage import io
from PIL import Image

CLASSES ={
    0: "background",
    1: "forest",
    2: "water",
    3: "agricultural",
    4: "residential,commercial,industrial",
    5: "grassland,swamp,shrubbery",
    6: "railway,trainstation",
    7: "highway,squares",
    8: "airport,shipyard",
    9: "roads",
    10: "buildings"
}

train_list = (
    'aachen',
    'bergisch',
    'bielefeld',
    'bochum',
    'bonn',
    'borken',
    'bottrop',
    'coesfeld',
    'dortmund',
    'dueren',
    'duisburg',
    'ennepetal',
    'erftstadt',
    'essen',
    'euskirchen',
    'gelsenkirchen',
    'guetersloh',
    'hagen',
    'hamm',
    'heinsberg',
    'herford',
    'hoexter',
    'kleve',
    'koeln',
    'krefeld',
    'leverkusen',
    'lippetal',
    'lippstadt',
    'lotte',
    'moenchengladbach',
    'moers',
    'muelheim',
    'muenster',
    'oberhausen',
    'paderborn',
    'recklinghausen',
    'remscheid',
    'siegen',
    'solingen',
    'wuppertal',
)

test_list = ('duesseldorf', 'herne', 'neuss')

class DataClass(torch.utils.data.Dataset):
    def __init__(self, root, split):
        super(DataClass, self).__init__()
        self.root = root
        self.split = split

        # self.image_path_list = glob(os.path.join(root, "images", "*", "*.jpg"))
        self.city_names = train_list if split=="train" else test_list
        file_list = []
        for cn in self.city_names:
            pattern = os.path.join(self.root, cn, "*rgb.jp2")
            file_list.extend(glob(pattern))
        self.image_path_list = sorted(file_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        mask_path = image_path.replace("_rgb.jp2", "_seg.tif")

        height_path = image_path.replace("_rgb.jp2", "_dem.tif")

        image_name = os.path.basename(image_path)

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        mask = io.imread(mask_path)
        height = io.imread(height_path)
        image = io.imread(image_path)
        return {
            "name": image_name, 
            "image": image, 
            "class": mask,
            "height": height
        }

    def __len__(self):
        return len(self.image_path_list)

# Define the process function at the top level
def process_sample_fn(dataset, index):
    """
    Process a single sample from the So2Sat dataset and return LitData-compatible data.
    """
    sample = dataset[index]
    # Convert the image to JPEG

    ##########Change this accoring to the dims of images.
    image = sample["image"].astype(np.uint8)  # Convert to HWC 
    mask = sample["class"].astype(np.uint8)
    height = sample["height"].astype(np.float32)
    # image = sample["image"].astype(np.uint8)  # Convert to HWC 
    # Convert class label to bytes

    out = {
        "name": sample["name"],
        "image": image,
        "class": mask,
        "height": height
    }

    return out

# Create a wrapper for picklability
class ProcessSampleWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, index):
        return process_sample_fn(self.dataset, index)

if __name__ == "__main__":
    ########## Change the paths!
    root = "/data/chen/datasets/GeoNRW/"
    output_folder = "/data/chen/datasets/outputs/GeoNRW"

    os.makedirs(root, exist_ok=True)

    for split, split_name in zip(["train", "test"], ["train", "test"]):
        output_dir = os.path.join(output_folder, split)
        os.makedirs(output_dir, exist_ok=True)

        output_flag = os.path.join(output_folder, split+".finish")
        if os.path.exists(output_flag):
            continue

        dataset = DataClass(root=root, split=split_name)

        ########## Change Metadata!!!
        metadata = {
            "Dataset": "GeoNRW",
            "split": split,
            "num_samples": len(dataset),
            "attributes": {
                "name": {"dtype": "str"},
                "image": {"dtype": "uint8", "format": "numpy"},
                "class": {"dtype": "uint8", "format": "numpy"},
                "height": {"dtype": "float32", "format": "numpy"},
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

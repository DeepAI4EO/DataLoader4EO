import numpy as np
from torchgeo.datasets import So2Sat as DataClass
import dataset4eo as eodata
import json
import os
from huggingface_hub import HfApi

# Define the process function at the top level
def process_sample_fn(dataset, index):
    """
    Process a single sample from the So2Sat dataset and return LitData-compatible data.
    """
    sample = dataset[index]
    # Convert the image to JPEG
    image = sample["image"].numpy().transpose(1, 2, 0).astype(np.float16)  # Convert to HWC
    # Convert class label to bytes
    class_data = sample["label"].numpy().astype(np.uint8)

    return {
        "image": image,
        "class": class_data,
    }

# Create a wrapper for picklability
class ProcessSampleWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, index):
        return process_sample_fn(self.dataset, index)

if __name__ == "__main__":
    root = "/data/chen/datasets/So2Sat/"
    output_folder = "./outputs/So2Sat"
    for split in ["train", "validation", "test"]:
        output_dir = os.path.join(output_folder, split)
        os.makedirs(output_dir, exist_ok=True)

        output_flag = os.path.join(os.path.dirname(os.path.abspath(__file__)), split+".finish")
        if os.path.exists(output_flag):
            continue

        dataset = DataClass(root=root, split=split)

        metadata = {
            "Dataset": "So2Sat",
            "split": split,
            "num_samples": len(dataset),
            "attributes": {
                "name": {"dtype": "str"},
                "image": {"dtype": "float16", "format": "numpy"},
                "class": {"dtype": "uint8", "format": "numpy"},
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

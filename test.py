import numpy as np
from PIL import Image
import litdata as ld

def random_images(index):
    fake_images = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    fake_labels = np.random.randint(10)

    # You can use any key:value pairs. Note that their types must not change between samples, and Python lists must
    # always contain the same number of elements with the same types.
    data = {"index": index, "image": fake_images, "class": fake_labels}

    return data

if __name__ == "__main__":
    # The optimize function writes data in an optimized format.
    ld.optimize(
        fn=random_images,                   # the function applied to each input
        inputs=list(range(1000)),           # the inputs to the function (here it's a list of numbers)
        output_dir="fast_data",             # optimized data is stored here
        num_workers=4,                      # The number of workers on the same machine
        chunk_bytes="64MB"                  # size of each chunk
    )
import dataset4eo as eodata
import litdata as ld
import time
from huggingface_hub import snapshot_download


repo_id = eodata.builtin_datasets['so2sat']

local_path = snapshot_download(
    repo_id=repo_id,
    repo_type = "dataset",
    cache_dir="./data_so2sat_cls",  # Custom directory
    revision="main"                 # Specific branch, tag, or commit
)

split = "train"

train_dataset = eodata.StreamingDataset(input_dir=f"{local_path}/{split}", num_channels=18, channels_to_select=[0, 3, 5, 7, 9], shuffle=True, drop_last=True)
train_dataloader = ld.StreamingDataLoader(train_dataset)

iters = 0
start = time.time()
for sample in train_dataloader:
    img, cls = sample['image'], sample['class']
    iters += 1
    if iters == 100:
        break
    
end = time.time()
print(end-start)

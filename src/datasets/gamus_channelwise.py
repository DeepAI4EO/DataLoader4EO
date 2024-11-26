import litdata as ld
import time
from multichannel import ChannelwiseDataset

train_dataset2 = ChannelwiseDataset('./optimized_gamus_dataset', channels_to_select=[0,1,2], other_keys=["height_map"], shuffle=True, drop_last=True)


train_dataloader = ld.StreamingDataLoader(train_dataset2)

iters = 0
start = time.time()
for sample in train_dataloader:
    img, cls, height = sample['image'], sample['segmentation_map'], sample["height_map"]
    iters += 1
    if iters == 100:
        break
end = time.time()
print(end-start)

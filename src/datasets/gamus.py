import litdata as ld
from torchvision import transforms
import time

train_dataset = ld.StreamingDataset('./optimized_gamus_dataset2', shuffle=True, drop_last=True)

train_dataloader = ld.StreamingDataLoader(train_dataset)

iters = 0

start = time.time()
for sample in train_dataloader:
    base_name, img, cls, height = sample["name"], sample['image'], sample['class'], sample["height"]
    iters += 1
    if iters == 100:
        break
end = time.time()
print(end-start)
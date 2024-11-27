import pdb
import litdata as ld
from torchvision import transforms
import time

train_dataset = ld.StreamingDataset('./gamus_dataset_val', shuffle=True, drop_last=True)
pdb.set_trace()


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
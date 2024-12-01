import dataset4eo as eodata
import litdata as ld #type: ignore
import time


train_dataset = eodata.StreamingDataset('./gamus_dataset_val', num_channels=3, shuffle=True, drop_last=True)
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
import dataset4eo as eodata
import litdata as ld
import time

train_dataset = eodata.StreamingDataset(input_dir="/data/chen/dataset4eo/outputs/So2Sat/train", num_channels=18, channels_to_select=[0, 3, 5, 7, 9], shuffle=True, drop_last=True)
train_dataloader = ld.StreamingDataLoader(train_dataset)

iters = 0
start = time.time()
for sample in train_dataloader:
    img, cls = sample['image'], sample['class']
    print(img.shape, cls)
    iters += 1
    if iters == 100:
        break
    
end = time.time()
print(end-start)
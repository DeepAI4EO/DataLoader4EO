import litdata as ld
from torchvision import transforms

transform = transforms.Compose([
        transforms.ToTensor(),
    ])
train_dataset = ld.StreamingDataset('./optimized_gamus_dataset', shuffle=True, drop_last=True)

train_dataloader = ld.StreamingDataLoader(train_dataset)

for sample in train_dataloader:
    base_name, img, cls, height = sample["name"], sample['image'], sample['class'], sample["height"]
    print(base_name)
    print(img.size())
    print(cls.size())
    print(height.shape)
    break
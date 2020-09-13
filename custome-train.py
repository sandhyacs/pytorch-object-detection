from detecto import core, utils, visualize
from torchvision import transforms


augmentations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(saturation=0.5),
    transforms.ToTensor(),
    utils.normalize_transform(),
])


val_dataset = core.Dataset('/dataset/validation_images/')
dataset = core.Dataset('//dataset/train_images/', transform=augmentations)
model = core.Model(['rust'])
loader = core.DataLoader(dataset, batch_size=2, shuffle=True)

losses = model.fit(loader, val_dataset, epochs=10, learning_rate=0.001,
                   lr_step_size=5, verbose=True)

model.save('model/model_weights.pth')

# from tqdm import tqdm
# import torch
# import torch.nn as nn

# from torchvision.datasets import CelebA, MNIST
# from torchvision import transforms
# from torchvision.transforms.functional import to_pil_image
# from torch.utils.data import DataLoader


# from src.model import RefineNet
# from src.score_matching import linear_noise_scale, score_matching_loss

# torch.cuda.empty_cache()


# '''train_dataset = CelebA(root='../../data', split='train', download=True,
#                           transform=transforms.Compose([
#                                 transforms.CenterCrop(140),
#                                 transforms.Resize(32),
#                                 transforms.ToTensor()
#                             ]))'''


# bsz = 16
# device = torch.device('cuda:1')
# noise_steps = 10


# train_dataset = MNIST(root='../../data', train=True, download=True, transform=transforms.ToTensor())
# train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=8, drop_last=True)
# model = RefineNet(
#     in_channels=1,
#     hidden_channels=(64, 128, 256, 512),
#     n_noise_scale=noise_steps
# ).to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# noise_scales = linear_noise_scale(start=1., end=0.01, length=10).to(device)


# for epoch in range(100):z
#     print(f'Epoch {epoch}')
#     epoch_loss = 0.
#     for i, (x, _) in enumerate(tqdm(train_dataloader)):
#         optimizer.zero_grad()
#         x = x.to(device)
#         loss = score_matching_loss(model, x, noise_scales)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     print(f'Epoch loss: {epoch_loss / len(train_dataloader)}')
# torch.save(model.state_dict(), 'ckpts/refinenet_mnist.pth')




from tqdm import tqdm
import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, Dataset

from src.model import RefineNet
from src.score_matching import linear_noise_scale, score_matching_loss

torch.cuda.empty_cache()

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label, you can modify this if you have labels

bsz = 16
device = torch.device('cuda:1')
noise_steps = 10

root_dir = './data/2DDesignWheelDataset'  # path to your custom image dataset
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

train_dataset = CustomImageDataset(root_dir, transform)
train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=15, drop_last=True)

model = RefineNet(
    in_channels=1,
    hidden_channels=(64, 128, 256, 512),
    n_noise_scale=noise_steps
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
noise_scales = linear_noise_scale(start=1., end=0.01, length=10).to(device)

for epoch in range(5):
    print(f'Epoch {epoch}')
    epoch_loss = 0.
    for i, (x, _) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        x = x.to(device)
        loss = score_matching_loss(model, x, noise_scales)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch loss: {epoch_loss / len(train_dataloader)}')

# Save the trained model
torch.save(model.state_dict(), 'ckpts/refinenet_custom.pth')

"""
Extremely Minimalistic Implementation of DDPM

https://arxiv.org/abs/2006.11239

Everything is self contained. (Except for pytorch and torchvision... of course)

run it with `python superminddpm.py`
"""

from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torchvision.datasets import ImageFolder  
from PIL import Image


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)


class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = ImageFolder(self.root_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, _ = self.dataset.imgs[idx]  # Discard labels
        image = self.dataset.loader(image_path)
        image = image.convert('L')
        if self.transform:
            image = self.transform(image)

        return image,0  # Return only the image

class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i
    

# class UNet(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, depth: int = 4):
#         super().__init__()

#         self.down_convs = nn.ModuleList()
#         self.up_convs = nn.ModuleList()

#         # Down-sampling (Encoder) Path
#         ch = in_channels  # Initial number of channels
#         for i in range(depth):
#             block = nn.Sequential(
#                 nn.Conv2d(ch, 2 * ch, 3, padding=1),
#                 nn.BatchNorm2d(2 * ch),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(2 * ch, 2 * ch, 3, padding=1),
#                 nn.BatchNorm2d(2 * ch),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2)
#             )
#             self.down_convs.append(block)
#             ch *= 2

#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(ch, 2 * ch, 3, padding=1),
#             nn.BatchNorm2d(2 * ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(2 * ch, ch, 3, padding=1),
#             nn.BatchNorm2d(ch),
#             nn.ReLU(inplace=True)
#         )

#         # Up-sampling (Decoder) Path
#         for i in range(depth - 1):
#             block = nn.Sequential(
#                 nn.ConvTranspose2d(2 * ch, ch // 2, kernel_size=2, stride=2),
#                 nn.Conv2d(ch, ch // 2, 3, padding=1),
#                 nn.BatchNorm2d(ch // 2),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(ch // 2, ch // 2, 3, padding=1),
#                 nn.BatchNorm2d(ch // 2),
#                 nn.ReLU(inplace=True),
#             )
#             self.up_convs.append(block)
#             ch //= 2

#         self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=1)

#     def forward(self, x, t):
#         skip_connections = []

#         for block in self.down_convs:
#             x = block(x)
#             skip_connections.append(x)

#         x = self.bottleneck(x)

#         for i, block in enumerate(self.up_convs):
#             skip = skip_connections.pop()  # Get the matching skip connection
#             x = torch.cat([skip, x], dim=1)  # Concatenate along channel dimension
#             x = block(x)

#         return self.out_conv(x)
    
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(blk(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(UpsampleBlock(feature*2, feature))
            self.ups.append(blk(feature*2, feature))
        
        self.bottleneck = blk(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], in_channels, kernel_size=1)

    def forward(self, x, t):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)



def train_mnist(n_epoch: int = 100, device="cuda") -> None:

    # ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    # eps_model = UNet(in_channels=1, out_channels=1)  # Instantiate U-Net with correct input/output channels
    # ddpm = DDPM(eps_model=eps_model, betas=(1e-4, 0.02), n_T=1000)
    ddpm = DDPM(eps_model=UNet(1), betas=(1e-4, 0.02), n_T=1000)
    ddpm.to(device)

    # tf = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    # )

    # dataset = MNIST(
    #     "./data",
    #     train=True,
    #     download=True,
    #     transform=tf,
    # )
    # tf = transforms.Compose(
    #     [
    #         transforms.Resize((64, 64)),
    #         transforms.Grayscale(),  # Convert to single-channel
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (1.0,)),  # Normalize with single mean/std
    #     ]
    # )

    tf = transforms.Compose([
        transforms.Resize((32, 32)),  # Adjust size accordingly
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,)),
    ])


    dataset = CustomImageDataset("./data", transform=tf) 
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=15)
    print(type(dataset))
    print(dataset[0])
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    for i in range(n_epoch):
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            # xh = ddpm.sample(16, (1, 28, 28), device)
            xh = ddpm.sample(16, (1, 32, 32), device)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ddpm_sample_{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")


if __name__ == "__main__":
    train_mnist()

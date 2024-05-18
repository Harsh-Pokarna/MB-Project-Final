import torch
from torchvision.transforms.functional import to_pil_image
from src.model import RefineNet
from src.score_matching import linear_noise_scale
from src.langevin_dynamics import sample


device = torch.device('cuda:1')
noise_steps = 10

model = RefineNet(
    in_channels=1,
    hidden_channels=(64, 128, 256, 512),
    n_noise_scale=noise_steps
).to(device)

model.load_state_dict(torch.load('./ckpts/refinenet_custom.pth'))
noise_scales = linear_noise_scale(start=1., end=0.01, length=10).to(device)

samples = sample(model, (2, 1, 32, 32), noise_scales, device)

samples = torch.clamp(samples, 0., 1.)
to_pil_image(samples[0])
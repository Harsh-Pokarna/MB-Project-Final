import torch
from torchvision.transforms.functional import to_pil_image
import os

from src.model import RefineNet
from src.score_matching import linear_noise_scale, score_matching_loss

torch.cuda.empty_cache()

device = torch.device('cuda:1')
# Assuming the model is defined as per the provided RefineNet class
model = RefineNet(in_channels=1, hidden_channels=(64, 128, 256, 512), n_noise_scale=10)
model.load_state_dict(torch.load('ckpts/refinenet_custom.pth'))  # Load trained model weights
model.to(device)
model.eval()

# Generate images
sampled_images = []
with torch.no_grad():
    for _ in range(50):  # Generate 50 images
        z = torch.randn((1, 1, 32, 32), device=device)  # Adjust size according to the model's first layer input
        noise_scale_idx = model.n_noise_scale - 1  # Use the last (smallest) noise scale
        sampled_image = model(z, noise_scale_idx)
        sampled_images.append(to_pil_image(sampled_image.squeeze(0).cpu()))  # Convert to PIL Image

# Ensure directory exists
os.makedirs('generated_images', exist_ok=True)

# Save the generated images
for idx, img in enumerate(sampled_images):
    img.save(f'generated_images/sample_{idx}.png')

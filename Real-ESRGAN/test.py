import torch
import os
checkpoint = torch.load( os.path.join(os.path.dirname(__file__), 'weights', 'RealESRGAN_x4plus.pth'), map_location="cpu")
print(checkpoint.keys())
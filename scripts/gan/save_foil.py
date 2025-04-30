#This script generates airfoils using the GAN and saves them in .npy.

import torch
import numpy as np
from models import Generator
from airbezier import bezier, normalise_airfoil

device = 'cuda' if torch.cuda.is_available() else 'cpu'
z_dim = 128
num_samples = 10000

gen = Generator(z_dim=z_dim).to(device)
gen.load_state_dict(torch.load(r'AirfoilLab\airfoil_gan\generator.pth', map_location=device))
gen.eval()

z = torch.randn(num_samples, z_dim).to(device)
with torch.no_grad():
    generated = gen(z).cpu().numpy()

processed = []
for airfoil in generated:
    norm = normalise_airfoil(airfoil)
    smooth = bezier(norm)
    processed.append(smooth)

processed = np.array(processed)
np.save(r'AirfoilLab\airfoil_gan\generated_airfoils_bezier.npy', processed)
print(f"Saved {processed.shape[0]} airfoils to 'generated_airfoils_bezier.npy'")

import torch
import matplotlib.pyplot as plt
import numpy as np
from models import Generator
from airbezier import bezier, normalise_airfoil

import scienceplots
import argparse

plt.style.use('science')

# Function to plot the generated airfoils
def main(mode='bezier', num_samples=10, z_dim=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = Generator(z_dim=z_dim).to(device)
    gen.load_state_dict(torch.load(r'AirfoilLab\airfoil_gan\generator.pth', map_location=device))
    gen.eval()

    z = torch.randn(num_samples, z_dim).to(device)
    with torch.no_grad():
        generated = gen(z).cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    axes = axes.flatten()

    for i, airfoil in enumerate(generated):
        ax = axes[i]
        norm = normalise_airfoil(airfoil)
        show_raw = mode in ('raw', 'raw+bezier')
        show_bez = mode in ('bezier', 'raw+bezier')
        if show_raw:
            ax.plot(norm[:, 0], norm[:, 1], 'o', alpha=0.3, label='Raw')
        if show_bez:
            smoothed = bezier(norm)
            ax.plot(smoothed[:, 0], smoothed[:, 1], '-', lw=2, label='Bezier')
        ax.set_title(f"Airfoil {i+1}", fontsize=10)
        ax.invert_yaxis()
        ax.set_aspect('equal')

        all_x = norm[:, 0]
        all_y = norm[:, 1]
        x_margin, y_margin = 0.05, 0.05
        ax.set_xlim([all_x.min() - x_margin, all_x.max() + x_margin])
        ax.set_ylim([all_y.max() + y_margin, all_y.min() - y_margin])
        ax.tick_params(labelsize=8)

        if i == 4:
            ax.legend(fontsize=8, frameon=False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['raw', 'bezier', 'raw+bezier'], default='bezier')
    args = parser.parse_args()
    main(mode=args.mode)

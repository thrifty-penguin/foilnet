import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import AirfoilDataset
from models import Generator, Critic
from gp import gradient_penalty

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Setup
z_dim = 128     # Latent space dimension
lambda_gp = 10  # Gradient penalty lambda hyperparameter
n_critic = 1    # Number of critic iterations per generator iteration
batch_size = 64 # Batch size
epochs = 10000  # Number of epochs to train for   

# Load data
dataset = AirfoilDataset(r"data\airfoil_interp.npy")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Init models and optimizers
gen = Generator(z_dim).to(device)
crit = Critic().to(device)
opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_crit = optim.Adam(crit.parameters(), lr=1e-4, betas=(0.0, 0.9))

# Training loop
for epoch in range(epochs):
    for batch_idx, real in enumerate(dataloader):
        real = real.to(device)

        # Train Critic
        for _ in range(n_critic):
            z = torch.randn(real.size(0), z_dim).to(device)
            fake = gen(z).detach()
            critic_real = crit(real)
            critic_fake = crit(fake)
            gp = gradient_penalty(crit, real, fake, device)

            loss_crit = -(critic_real.mean() - critic_fake.mean()) + lambda_gp * gp
            opt_crit.zero_grad()
            loss_crit.backward()
            opt_crit.step()

        # Train Generator
        z = torch.randn(real.size(0), z_dim).to(device)
        fake = gen(z)
        loss_gen = -crit(fake).mean()
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print every batch
    print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}: D_loss={loss_crit.item():.4f}, G_loss={loss_gen.item():.4f}")

#save models
torch.save(gen.state_dict(), r'models\generator.pth')
torch.save(crit.state_dict(), r'models\critic.pth')
print("Models saved as generator.pth and critic.pth in the models directory.")
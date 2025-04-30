import torch

# This function computes the gradient penalty for the Wasserstein GAN with gradient penalty (WGAN-GP) loss.
def gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1).to(device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    critic_score = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_score),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1)**2).mean()
    return gp

import torch


def add_noise(
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    total_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add simple timestep-scaled Gaussian noise and return the noise target."""

    noise = torch.randn_like(clean)
    alpha = timesteps.float() / float(total_timesteps)
    while alpha.ndim < clean.ndim:
        alpha = alpha.unsqueeze(-1)
    return clean + noise * alpha, noise

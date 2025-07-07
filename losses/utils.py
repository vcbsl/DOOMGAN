import torch

def gradient_penalty(netD, real_data, fake_data, heatmaps, device):
    """Calculates the gradient penalty loss for WGAN-GP."""
    b_size = real_data.size(0)
    alpha = torch.rand(b_size, 1, 1, 1, device=device)
    inter_img = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    inter_hm = heatmaps.requires_grad_(True) # Use same heatmaps for both

    disc_inter = netD(inter_img, inter_hm)[-1]
    grads = torch.autograd.grad(
        outputs=disc_inter, inputs=[inter_img, inter_hm],
        grad_outputs=torch.ones_like(disc_inter, device=device),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grads = grads.view(b_size, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()
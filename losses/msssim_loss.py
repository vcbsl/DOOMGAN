import torch
import torch.nn.functional as F
from torch.nn import Module

# Code adapted from https://github.com/jorge-pessoa/pytorch-msssim
# A faithful PyTorch port of the official TensorFlow implementation.
class MSSSIMLoss(Module):
    def __init__(self, window_size=11, size_average=True, channel=3, device='cpu'):
        super(MSSSIMLoss, self).__init__()
        self.window_size, self.size_average, self.channel = window_size, size_average, channel
        self.device = device
        self.weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device)
        self.window = self._create_window(self.window_size, self.channel).to(self.device)

    def _gaussian(self, window_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)]))
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel=1):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
        return ssim_map.mean() if size_average else ssim_map.mean([1,2,3])

    def forward(self, img1, img2):
        levels = self.weights.size(0)
        mssim = []
        for _ in range(levels):
            ssim_val = self._ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)
            mssim.append(ssim_val)
            img1, img2 = [F.avg_pool2d(x, (2, 2)) for x in [img1, img2]]
        mssim = torch.stack(mssim)
        mssim_pow = torch.pow(mssim.clamp(min=1e-8), self.weights)
        return 1.0 - torch.prod(mssim_pow)
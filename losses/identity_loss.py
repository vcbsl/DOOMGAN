import torch
import torch.nn as nn

class IdentityLoss(nn.Module):
    """Calculates the average cosine distance loss: 1 - cos_sim(E1, E2)"""
    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, real_embeddings, fake_embeddings):
        return (1 - self.cosine_similarity(real_embeddings, fake_embeddings)).mean()

class IdentityDifferenceLoss(nn.Module):
    """Calculates the absolute difference: |1 - cos_sim(E1, E2)|"""
    def __init__(self):
        super(IdentityDifferenceLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, real_embeddings, fake_embeddings):
        return torch.abs(1 - self.cosine_similarity(real_embeddings, fake_embeddings)).mean()
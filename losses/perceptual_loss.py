import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    Calculates the VGG-based perceptual loss.

    Uses features from the VGG19 network to compute the L1 loss between
    the feature maps of the predicted and target images.
    """
    def __init__(self, device, loss_layers=[3, 8, 17, 26, 35]):
        """
        Initializes the PerceptualLoss module.

        Args:
            device: The device (e.g., 'cuda:0' or 'cpu') to run the model on.
            loss_layers (list, optional): A list of VGG19 layer indices to use for feature extraction.
        """
        super(PerceptualLoss, self).__init__()
        
        # Load the pre-trained VGG19 model
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        
        # Disable inplace operations in VGG19's ReLU layers to prevent modification of the input tensor during the forward pass,
        # which would otherwise cause a RuntimeError in the backward pass.
        for module in vgg.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        # Freeze the VGG model parameters
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Extract the desired layers for the loss calculation
        self.vgg_layers = nn.ModuleList([vgg[i] for i in loss_layers])
        
        # Define the normalization constants for ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _get_features(self, x):
        """
        Extracts VGG features from an image tensor.
        The input image is expected to be in the range [-1, 1].
        """
        # Rescale images from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Normalize images according to ImageNet standards
        x = (x - self.mean) / self.std
        
        features = []
        for layer in self.vgg_layers:
            x = layer(x)
            features.append(x)
        return features

    def forward(self, pred, target):
        """
        Computes the perceptual loss between a predicted image and a target image.
        """
        pred_feats = self._get_features(pred)
        target_feats = self._get_features(target)
        
        # Calculate the L1 loss between the feature maps
        loss = sum(torch.nn.functional.l1_loss(pf, tf) for pf, tf in zip(pred_feats, target_feats))
        
        return loss
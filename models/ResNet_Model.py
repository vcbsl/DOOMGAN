import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50_ArcFace(nn.Module):
    """
    ResNet-50 model modified for ArcFace loss.
    Outputs embeddings that can be used for feature extraction.
    """
    def __init__(self, embedding_size=512, pretrained=True):
        super(ResNet50_ArcFace, self).__init__()
        self.embedding_size = embedding_size

        # Load a pre-trained ResNet-50 model
        self.backbone = models.resnet50(pretrained=pretrained)

        # Modify the final fully connected layer
        # Replace the last fully connected layer with a linear layer to get embeddings
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, self.embedding_size)

        # Normalize the embedding vectors
        self.l2_norm = nn.functional.normalize

    def forward(self, x):
        x = self.backbone(x)
        # Normalize embeddings to have unit length
        x = self.l2_norm(x, p=2, dim=1)
        return x

# Example usage
if __name__ == "__main__":
    # Load config parameters if needed
    import yaml
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    model = ResNet50_ArcFace(
        embedding_size=config['embedding_size'],
        pretrained=True
    ).to(device)

    # Print model architecture
    print(model)

    # Test with a random input
    dummy_input = torch.randn(1, 3, config['image_height'], config['image_width']).to(device)
    embeddings = model(dummy_input)
    print("Embeddings shape:", embeddings.shape)
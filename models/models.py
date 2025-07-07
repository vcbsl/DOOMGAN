import torch
import torch.nn as nn

def weights_init(m):
    """
    Applies custom weights initialization to a model's modules.
    - Conv layers: He (Kaiming) normal initialization.
    - InstanceNorm layers: Normal distribution for weights, constant for biases.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Use a fan-in He initialization for Conv layers
        nn.init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in')
    elif classname.find('InstanceNorm') != -1:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        attention = self.softmax(torch.bmm(proj_query, proj_key))
        proj_value = self.value(x).view(B, -1, W * H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, nc=3, ndf=64, nz=200, num_landmarks=19):
        super(Encoder, self).__init__()
        in_channels = nc + num_landmarks
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2),
            ResidualBlock(ndf*8), SelfAttention(ndf*8),
            nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf*16), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf*16, ndf*16, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 16, nz, 1)
        )
    def forward(self, img, heatmaps):
        return self.model(torch.cat([img, heatmaps], 1)).view(img.size(0), -1)

class Generator(nn.Module):
    def __init__(self, nz=200, ngf=64, nc=3, landmark_feature_size=128):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.fc = nn.Sequential(
             nn.Linear(nz + landmark_feature_size, ngf * 32 * 4 * 4),
             nn.ReLU() # Removed inplace=True
        )
        def block(in_c, out_c):
            return [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False), nn.InstanceNorm2d(out_c), nn.ReLU()]
        self.main = nn.Sequential(
            ResidualBlock(ngf * 32), # 4x4
            *block(ngf*32, ngf*16), # 8x8
            *block(ngf*16, ngf*8), # 16x16
            SelfAttention(ngf * 8),
            *block(ngf*8, ngf*4), # 32x32
            *block(ngf*4, ngf*2), # 64x64
            *block(ngf*2, ngf), # 128x128
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1), nn.Tanh() # 256x256
        )
    def forward(self, z, landmark_features):
        x = self.fc(torch.cat([z, landmark_features], 1)).view(-1, self.ngf*32, 4, 4)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, num_landmarks=19):
        super(Discriminator, self).__init__()
        in_channels = nc + num_landmarks
        def block(in_c, out_c, norm=True, dropout=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1)) if norm else nn.Conv2d(in_c, out_c, 4, 2, 1)]
            layers.append(nn.LeakyReLU(0.2)) 
            if dropout: layers.append(nn.Dropout(0.5))
            layers.append(ResidualBlock(out_c))
            return layers
        self.model = nn.ModuleList([
            nn.Sequential(*block(in_channels, ndf, norm=False, dropout=False)), # 128
            nn.Sequential(*block(ndf, ndf * 2)), # 64
            nn.Sequential(*block(ndf*2, ndf * 4)), # 32
            nn.Sequential(SelfAttention(ndf*4), *block(ndf*4, ndf * 8)), # 16
            nn.Sequential(*block(ndf*8, ndf * 16)), # 8
        ])
        self.out_layers = nn.ModuleList([
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, 1, 3, 1, 1)),
            nn.utils.spectral_norm(nn.Conv2d(ndf*4, 1, 3, 1, 1)),
            nn.utils.spectral_norm(nn.Conv2d(ndf*8, 1, 3, 1, 1)),
            nn.utils.spectral_norm(nn.Conv2d(ndf*16, 1, 3, 1, 1)),
        ])
    def forward(self, img, heatmaps):
        x = torch.cat([img, heatmaps], 1)
        outputs = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i > 0: outputs.append(self.out_layers[i-1](x))
        return outputs

class LandmarkEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LandmarkEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2), 
            nn.Linear(128, output_dim), nn.LeakyReLU(0.2)
        )
    def forward(self, landmarks):
        return self.encoder(landmarks)
import torch
from models import Generator, Discriminator, Encoder
from utils import create_landmark_heatmaps

def test_models():
    """Runs a series of tests to check the forward pass of all models."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on {device}.")

    # --- Config ---
    cfg = {'b': 4, 'nz': 200, 'ngf': 64, 'ndf': 64, 'nc': 3, 'lfs': 128, 'isz': 256, 'nl': 19}

    # --- Test Generator ---
    print("\n--- Testing Generator ---")
    z = torch.randn(cfg['b'], cfg['nz'], device=device)
    lf = torch.randn(cfg['b'], cfg['lfs'], device=device)
    netG = Generator(cfg['nz'], cfg['ngf'], cfg['nc'], cfg['lfs']).to(device)
    out = netG(z, lf)
    assert out.shape == (cfg['b'], cfg['nc'], cfg['isz'], cfg['isz']), f"G output shape is {out.shape}"
    print("Generator test PASSED!")

    # --- Test Discriminator ---
    print("\n--- Testing Discriminator ---")
    img = torch.randn(cfg['b'], cfg['nc'], cfg['isz'], cfg['isz'], device=device)
    lndm = torch.rand(cfg['b'], cfg['nl'] * 2, device=device)
    hmaps = create_landmark_heatmaps(lndm, cfg['isz']).to(device)
    netD = Discriminator(cfg['nc'], cfg['ndf'], num_landmarks=cfg['nl']).to(device)
    out = netD(img, hmaps)
    assert len(out) == 4, f"D output length is {len(out)}"
    print("Discriminator test PASSED!")

    # --- Test Encoder ---
    print("\n--- Testing Encoder ---")
    netE = Encoder(cfg['nc'], cfg['ndf'], cfg['nz'], num_landmarks=cfg['nl']).to(device)
    out = netE(img, hmaps)
    assert out.shape == (cfg['b'], cfg['nz']), f"E output shape is {out.shape}"
    print("Encoder test PASSED!")

if __name__ == '__main__':
    test_models()

"""
To run the tests: python -m models.test_models

"""
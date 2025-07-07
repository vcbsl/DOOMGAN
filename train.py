import os
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import wandb
import warnings
from torchvision.utils import save_image

# Local imports
from dataset import get_dataloader
from models import Generator, Discriminator, Encoder, LandmarkEncoder, weights_init, ResNet50_ArcFace
from losses import PerceptualLoss, MSSSIMLoss, IdentityLoss, IdentityDifferenceLoss, gradient_penalty
from utils import create_landmark_heatmaps, DynamicWeightAdjuster

ARCFACE_CLASS_AVAILABLE = ResNet50_ArcFace is not None
if not ARCFACE_CLASS_AVAILABLE:
    warnings.warn("ResNet_Model.py or its dependencies not found. Identity losses will be disabled.")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_environment(config):
    seed = config['manual_seed']
    random.seed(seed)
    torch.manual_seed(seed)
    if config.get('use_deterministic_algorithms', False):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    wandb.init(project=config['project_name'], config=config)
    device = torch.device(config['device'] if torch.cuda.is_available() and config['ngpu'] > 0 else "cpu")
    print(f"Using device: {device}")
    return device

def main():
    config = load_config('config/config.yaml')
    device = setup_environment(config)
    
    dataloader = get_dataloader(config)
    model_cfg = config['model']
    data_cfg = config['data']

    models = {
        'G': Generator(nz=model_cfg['nz'], ngf=model_cfg['ngf'], nc=data_cfg['nc'], landmark_feature_size=model_cfg['landmark_feature_size']),
        'D': Discriminator(nc=data_cfg['nc'], ndf=model_cfg['ndf'], num_landmarks=model_cfg['num_landmarks']),
        'E': Encoder(nc=data_cfg['nc'], ndf=model_cfg['ndf'], nz=model_cfg['nz'], num_landmarks=model_cfg['num_landmarks']),
        'LE': LandmarkEncoder(input_dim=model_cfg['num_landmarks'] * 2, output_dim=model_cfg['landmark_feature_size'])
    }

    for name, model in models.items():
        model.to(device).apply(weights_init)
        if device.type == 'cuda' and config['ngpu'] > 1:
            models[name] = nn.DataParallel(model, list(range(config['ngpu'])))

    losses = {
        'perceptual': PerceptualLoss(device=device),
        'msssim': MSSSIMLoss(channel=data_cfg['nc'], device=device)
    }
    
    arcface_loaded_successfully = False
    embedding_model = None
    if ARCFACE_CLASS_AVAILABLE:
        try:
            arcface_path = config['paths']['inputs']['arcface_model']
            embedding_model = ResNet50_ArcFace(embedding_size=512, pretrained=False).to(device)
            checkpoint = torch.load(arcface_path, map_location=device)
            embedding_model.load_state_dict(checkpoint['model_state_dict'])
            embedding_model.eval()
            for param in embedding_model.parameters(): param.requires_grad = False
            losses['identity'] = IdentityLoss()
            losses['identity_diff'] = IdentityDifferenceLoss()
            arcface_loaded_successfully = True
            print("ArcFace model loaded successfully for identity losses.")
        except FileNotFoundError:
            warnings.warn(f"ArcFace model weights not found at {arcface_path}. Disabling identity losses.")
        except Exception as e:
            warnings.warn(f"An error occurred while loading ArcFace model: {e}. Disabling identity losses.")

    opt_cfg = config['training']['optimizer']
    optimizers = {
        'D': AdamW(models['D'].parameters(), lr=opt_cfg['lr_d'], betas=(opt_cfg['beta1'], 0.999), weight_decay=opt_cfg['weight_decay']),
        'G': AdamW(models['G'].parameters(), lr=opt_cfg['lr_g'], betas=(opt_cfg['beta1'], 0.999), weight_decay=opt_cfg['weight_decay']),
        'E': AdamW(models['E'].parameters(), lr=opt_cfg['lr_e'], betas=(opt_cfg['beta1'], 0.999), weight_decay=opt_cfg['weight_decay']),
        'LE': optim.Adam(models['LE'].parameters(), lr=opt_cfg['lr_le'], betas=(opt_cfg['beta1'], 0.999))
    }
    schedulers = {k: optim.lr_scheduler.ExponentialLR(v, config['training']['scheduler'][f'gamma_{k.lower()}']) for k, v in optimizers.items()}

    weight_adjuster = DynamicWeightAdjuster(config['training']['loss_weights']['initial_dynamic'])
    for path in config['paths']['outputs'].values():
        os.makedirs(path, exist_ok=True)

    print("Starting Training Loop...")
    for epoch in range(config['training']['num_epochs']):
        for i, (real_imgs, lndmks) in enumerate(dataloader):
            real_imgs, lndmks = real_imgs.to(device), lndmks.to(device)
            heatmaps = create_landmark_heatmaps(lndmks, data_cfg['image_size'])
            lndmk_features = models['LE'](lndmks)
            
            # --- Update D ---
            optimizers['D'].zero_grad()
            with torch.no_grad():
                noise_d = models['E'](real_imgs, heatmaps)
                fake_imgs_detached = models['G'](noise_d, lndmk_features)
            
            errD_real = sum(-torch.mean(o) for o in models['D'](real_imgs, heatmaps)) / 4
            errD_fake = sum(torch.mean(o) for o in models['D'](fake_imgs_detached, heatmaps)) / 4
            gp = gradient_penalty(models['D'], real_imgs, fake_imgs_detached, heatmaps, device)
            errD = errD_real + errD_fake + config['training']['loss_weights']['gp'] * gp
            errD.backward()
            optimizers['D'].step()

            # --- Update G, E, LE ---
            optimizers['G'].zero_grad(); optimizers['E'].zero_grad(); optimizers['LE'].zero_grad()
            noise_g = models['E'](real_imgs, heatmaps)
            fake_imgs = models['G'](noise_g, lndmk_features)
            outputs_fake_G = models['D'](fake_imgs, heatmaps)
            
            # Calculate and store original loss tensors ---
            loss_tensors = {}
            loss_tensors['base'] = sum(-torch.mean(o) for o in outputs_fake_G) / 4
            loss_tensors['reconstruction'] = nn.functional.l1_loss(fake_imgs, real_imgs)
            loss_tensors['perceptual'] = losses['perceptual'](fake_imgs, real_imgs)
            loss_tensors['ms_ssim'] = losses['msssim']((fake_imgs + 1) / 2, (real_imgs + 1) / 2)
            
            if arcface_loaded_successfully:
                with torch.no_grad():
                    real_embed = embedding_model(real_imgs)
                morph_embed = embedding_model(fake_imgs)
                loss_tensors['identity'] = losses['identity'](real_embed, morph_embed)
                loss_tensors['identity_diff'] = losses['identity_diff'](real_embed, morph_embed)

            # Create float values for weight adjuster and logging
            loss_values = {key: tensor.item() for key, tensor in loss_tensors.items()}
            updated_weights = weight_adjuster.update_weights(loss_values)
            
            # Calculate final loss from original tensors to preserve graph ---
            errG = sum(updated_weights[key] * loss_tensors[key] for key in loss_tensors)
            
            errG.backward()
            optimizers['G'].step(); optimizers['E'].step(); optimizers['LE'].step()
            
            # Logging
            if i % 50 == 0:
                print(f"[{epoch}/{config['training']['num_epochs']}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")
                log_data = {'epoch': epoch, 'iter': i, 'D_loss': errD.item(), 'G_loss': errG.item(), **{f'loss_{k}': v for k,v in loss_values.items()}, **{f'weight_{k}': v for k,v in updated_weights.items()}}
                wandb.log(log_data)
        
        # End of Epoch
        for scheduler in schedulers.values(): scheduler.step()
        if epoch > 0 and epoch % 10 == 0:
            for name, model in models.items():
                save_path = config['paths']['outputs'][name]
                torch.save(model.state_dict(), os.path.join(save_path, f'{name.lower()}_epoch_{epoch}.pth'))
            print(f"Saved models at epoch {epoch}")
            
            with torch.no_grad():
                fixed_noise = torch.randn(1, model_cfg['nz'], device=device)
                fixed_lndmks = lndmks[0].unsqueeze(0) if 'lndmks' in locals() and lndmks.numel() > 0 else torch.rand(1, model_cfg['num_landmarks'] * 2, device=device)
                fixed_lndmk_feats = models['LE'](fixed_lndmks)
                sample_img = models['G'](fixed_noise, fixed_lndmk_feats)
                save_image(sample_img, f"sample_epoch_{epoch}.png", normalize=True)
                wandb.log({"generated_samples": wandb.Image(f"sample_epoch_{epoch}.png")})

    wandb.finish()

if __name__ == '__main__':
    main()
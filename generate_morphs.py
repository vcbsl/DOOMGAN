import os
import yaml
import torch
import argparse
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import OrderedDict
from models import Generator, Encoder, LandmarkEncoder
from models.landmark_predictor import OcularLMGenerator

def remove_module_prefix(state_dict):
    """Removes the 'module.' prefix from state dict keys if it exists."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

def create_landmark_heatmaps(landmarks_tensor, image_size, sigma=4):
    """Generates Gaussian heatmaps from a NORMALIZED [0,1] landmark tensor."""
    batch_size, num_coords = landmarks_tensor.shape
    num_landmarks = num_coords // 2
    landmarks = landmarks_tensor.view(batch_size, num_landmarks, 2) * (image_size - 1)
    
    x = torch.arange(0, image_size, device=landmarks_tensor.device).float()
    y = torch.arange(0, image_size, device=landmarks_tensor.device).float()
    xx, yy = torch.meshgrid(y, x, indexing='ij')
    xx = xx.unsqueeze(0).unsqueeze(1).expand(batch_size, num_landmarks, -1, -1)
    yy = yy.unsqueeze(0).unsqueeze(1).expand(batch_size, num_landmarks, -1, -1)
    landmarks_x = landmarks[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    landmarks_y = landmarks[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    dist_sq = (xx - landmarks_y)**2 + (yy - landmarks_x)**2
    heatmaps = torch.exp(-dist_sq / (2 * sigma**2))
    return heatmaps

def load_all_models(config, epoch, device):
    """Loads all models with the correct dictionary keys from utils."""
    model_cfg = config['model']
    data_cfg = config['data']
    paths_cfg = config['paths']
    models = {}

    gan_model_classes = {
        'G': Generator(nz=model_cfg['nz'], ngf=model_cfg['ngf'], nc=data_cfg['nc'], landmark_feature_size=model_cfg['landmark_feature_size']),
        'E': Encoder(nc=data_cfg['nc'], ndf=model_cfg['ndf'], nz=model_cfg['nz'], num_landmarks=model_cfg['num_landmarks']),
        'LE': LandmarkEncoder(input_dim=model_cfg['num_landmarks'] * 2, output_dim=model_cfg['landmark_feature_size'])
    }
    for name, model in gan_model_classes.items():
        model_path = os.path.join(paths_cfg['outputs'][name], f'{name.lower()}_epoch_{epoch}.pth')
        print(f"Loading {name} model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(remove_module_prefix(state_dict))
        model.to(device).eval()
        if name == 'G': models['netG'] = model
        elif name == 'E': models['netE'] = model
        elif name == 'LE': models['landmark_encoder'] = model

    lp_path = paths_cfg['inputs']['landmark_predictor_model']
    print(f"Loading Landmark Predictor from: {lp_path}")
    landmark_predictor = OcularLMGenerator().to(device)
    state_dict_lp = torch.load(lp_path, map_location=device)
    landmark_predictor.load_state_dict(remove_module_prefix(state_dict_lp))
    landmark_predictor.eval()
    models['landmark_predictor'] = landmark_predictor

    return models

def predict_landmarks_for_gan(image, landmark_predictor_model, device):
    """The proven method to get high-quality landmarks."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_transformed = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        landmarks = landmark_predictor_model(image_transformed).squeeze(0).cpu()
        landmarks = landmarks[:38] # Explicitly take first 19 landmarks (38 coords)
        landmarks = landmarks.view(-1, 2)
        landmarks[:, 0] /= 256.0
        landmarks[:, 1] /= 256.0
        landmarks = landmarks.flatten()
        
    return landmarks.unsqueeze(0)

def process_single_image(image_pil, config, device, models):
    """Processes one image to get its GAN inputs (z and lf), using only the predictor."""
    # 1. Get Landmarks
    print("-> Predicting landmarks using model...")
    landmarks_tensor = predict_landmarks_for_gan(image_pil, models['landmark_predictor'], device).to(device)

    # 2. Transform image for the GAN
    image_tensor_gan = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])(image_pil).unsqueeze(0).to(device)

    # 3. Create Heatmap from the landmarks
    heatmap = create_landmark_heatmaps(landmarks_tensor, image_size=config['data']['image_size']).to(device)
    
    # 4. Encode the image and landmarks
    with torch.no_grad():
        lf = models['landmark_encoder'](landmarks_tensor)
        z = models['netE'](image_tensor_gan, heatmap)
    
    return z, lf

# --- MAIN SCRIPT EXECUTION ---
def main(args):
    """Main function to run the CLI morphing generation."""
    config = yaml.safe_load(open(args.config, 'r'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Interpolation Alpha: {args.alpha}")

    # 1. Load all models
    models = load_all_models(config, args.epoch, device)
    print("--- All models loaded successfully ---")

    # 2. Load source images
    image1_pil = Image.open(args.image1).convert('RGB')
    image2_pil = Image.open(args.image2).convert('RGB')

    # 3. Process each image using the single, reliable pipeline
    print("\n--- Processing Image 1 ---")
    z1, lf1 = process_single_image(image1_pil, config, device, models)
    print("\n--- Processing Image 2 ---")
    z2, lf2 = process_single_image(image2_pil, config, device, models)

    # 4. Interpolate and Generate
    print("\n--- Generating Morphed Image ---")
    alpha = args.alpha
    z_morph = (1 - alpha) * z1 + alpha * z2
    lf_morph = (1 - alpha) * lf1 + alpha * lf2
    
    with torch.no_grad():
        morphed_image_tensor = models['netG'](z_morph, lf_morph)
    
    # Denormalize for saving
    morphed_image = (morphed_image_tensor * 0.5 + 0.5).clamp(0, 1)
    morphed_image_numpy = morphed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 5. Save the output
    output_dir = os.path.dirname(args.output)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    
    final_image = Image.fromarray((morphed_image_numpy * 255).astype(np.uint8))
    final_image.save(args.output)
    print(f"\nMorphed image saved successfully to '{args.output}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a high-fidelity morphed ocular image using the DOOMGAN pipeline.")
    parser.add_argument('--image1', type=str, required=True, help='Path to the first source image.')
    parser.add_argument('--image2', type=str, required=True, help='Path to the second source image.')
    parser.add_argument('--epoch', type=int, required=True, help='The checkpoint epoch to load for GAN models.')
    parser.add_argument('--output', type=str, default='generated_morphs/morphed_image.png', help='Path to save the output morphed image.')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Interpolation factor: 0.0 for image1, 1.0 for image2. Default: 0.5.')
    
    args = parser.parse_args()
    main(args)
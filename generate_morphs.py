import os
import yaml
import json
import torch
import argparse
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

# Local imports from our project structure
from models import Generator, Encoder, LandmarkEncoder
from utils import create_landmark_heatmaps

def load_config(config_path='config/config.yaml'):
    """Loads the project configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_models(config, epoch, device):
    """Loads and initializes the Generator, Encoder, and LandmarkEncoder for a specific epoch."""
    model_cfg = config['model']
    data_cfg = config['data']
    paths_cfg = config['paths']['outputs']

    # Initialize models
    models = {
        'G': Generator(nz=model_cfg['nz'], ngf=model_cfg['ngf'], nc=data_cfg['nc'], landmark_feature_size=model_cfg['landmark_feature_size']),
        'E': Encoder(nc=data_cfg['nc'], ndf=model_cfg['ndf'], nz=model_cfg['nz'], num_landmarks=model_cfg['num_landmarks']),
        'LE': LandmarkEncoder(input_dim=model_cfg['num_landmarks'] * 2, output_dim=model_cfg['landmark_feature_size'])
    }
    
    # Helper to handle 'module.' prefix if model was saved with DataParallel
    def remove_module_prefix(state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        return new_state_dict

    # Load state dict for each model
    for name, model in models.items():
        model_path = os.path.join(paths_cfg[name], f'{name.lower()}_epoch_{epoch}.pth')
        print(f"Loading {name} model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        
    return models['G'], models['E'], models['LE']

def load_image_and_landmarks(image_path, landmarks_dict, config):
    """Loads a single image and its corresponding landmarks."""
    data_cfg = config['data']
    model_cfg = config['model']

    # 1. Load and transform the image
    transform = transforms.Compose([
        transforms.Resize((data_cfg['image_size'], data_cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension

    # 2. Load and process landmarks
    key = os.path.normpath(image_path).replace('\\', '/')
    
    landmarks_info = landmarks_dict.get(key)
    if landmarks_info is None:
        raise ValueError(f"Landmarks not found for image key: '{key}'")

    eye_landmarks = landmarks_info.get('Eye_landmarks', [])
    iris_landmarks = landmarks_info.get('Iris_landmarks', [])
    iris_center = landmarks_info.get('Iris_center', [])
    all_landmarks = eye_landmarks + iris_landmarks + iris_center

    if len(all_landmarks) < model_cfg['num_landmarks']:
        all_landmarks += [[0, 0]] * (model_cfg['num_landmarks'] - len(all_landmarks))
    else:
        all_landmarks = all_landmarks[:model_cfg['num_landmarks']]

    normalized_landmarks = [(x / data_cfg['image_size'], y / data_cfg['image_size']) for x, y in all_landmarks]
    landmarks_flat = [coord for point in normalized_landmarks for coord in point]
    landmarks_tensor = torch.tensor(landmarks_flat, dtype=torch.float32).unsqueeze(0) # Add batch dimension

    return image_tensor, landmarks_tensor

def main(args):
    """Main function to generate the morphed image."""
    config = load_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Models
    netG, netE, landmark_encoder = load_models(config, args.epoch, device)

    # 2. Load Data for both images
    with open(config['data']['landmark_json_path'], 'r') as f:
        landmarks_dict = json.load(f)

    img1, lndmks1 = load_image_and_landmarks(args.image1, landmarks_dict, config)
    img2, lndmks2 = load_image_and_landmarks(args.image2, landmarks_dict, config)
    
    img1, lndmks1 = img1.to(device), lndmks1.to(device)
    img2, lndmks2 = img2.to(device), lndmks2.to(device)

    # 3. Morphing Logic
    with torch.no_grad():
        # Create heatmaps
        heatmaps1 = create_landmark_heatmaps(lndmks1, config['data']['image_size'])
        heatmaps2 = create_landmark_heatmaps(lndmks2, config['data']['image_size'])
        
        # Encode images to latent vectors (z)
        z1 = netE(img1, heatmaps1)
        z2 = netE(img2, heatmaps2)
        
        # Encode landmarks to feature vectors (lf)
        lf1 = landmark_encoder(lndmks1)
        lf2 = landmark_encoder(lndmks2)

        # Interpolate in both latent space and landmark feature space
        z_morph = (z1 + z2) / 2.0
        lf_morph = (lf1 + lf2) / 2.0
        
        # Generate the morphed image
        morphed_image = netG(z_morph, lf_morph)

    # 4. Save and Display Results
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    save_image(morphed_image, args.output, normalize=True)
    print(f"Morphed image saved to '{args.output}'")
    
    if not args.no_display:
        # Denormalize for visualization
        def denormalize(tensor):
            return (tensor.cpu().squeeze(0) * 0.5) + 0.5
        
        img1_viz = denormalize(img1).permute(1, 2, 0).numpy()
        img2_viz = denormalize(img2).permute(1, 2, 0).numpy()
        morphed_viz = denormalize(morphed_image).permute(1, 2, 0).numpy()
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(img1_viz); plt.title("Subject 1"); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(morphed_viz); plt.title("Morphed"); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(img2_viz); plt.title("Subject 2"); plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a morphed ocular image from two source images.")
    parser.add_argument('--image1', type=str, required=True, help='Path to the first source image.')
    parser.add_argument('--image2', type=str, required=True, help='Path to the second source image.')
    parser.add_argument('--epoch', type=int, required=True, help='The checkpoint epoch to load.')
    parser.add_argument('--output', type=str, default='generated_morphs/morphed_image.png', help='Path to save the output morphed image.')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file.')
    parser.add_argument('--no-display', action='store_true', help='Do not display the images using matplotlib.')
    
    args = parser.parse_args()
    main(args)
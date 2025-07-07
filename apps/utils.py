import os
import torch
from collections import OrderedDict
from PIL import Image

# Local project imports
from models import Generator, Encoder, LandmarkEncoder
from models.landmark_predictor import OcularLMGenerator

def remove_module_prefix(state_dict):
    """Removes the 'module.' prefix from state dict keys if it exists."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

def load_all_models(config, epoch, device):
    """Loads and initializes all models needed for the Streamlit app."""
    model_cfg = config['model']
    data_cfg = config['data']
    paths_cfg = config['paths']

    # --- 1. Load the main GAN models (G, E, LE) ---
    gan_models = {
        'G': Generator(nz=model_cfg['nz'], ngf=model_cfg['ngf'], nc=data_cfg['nc'], landmark_feature_size=model_cfg['landmark_feature_size']),
        'E': Encoder(nc=data_cfg['nc'], ndf=model_cfg['ndf'], nz=model_cfg['nz'], num_landmarks=model_cfg['num_landmarks']),
        'LE': LandmarkEncoder(input_dim=model_cfg['num_landmarks'] * 2, output_dim=model_cfg['landmark_feature_size'])
    }
    for name, model in gan_models.items():
        model_path = os.path.join(paths_cfg['outputs'][name], f'{name.lower()}_epoch_{epoch}.pth')
        print(f"Loading {name} model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict)
        model.to(device).eval()

    # --- 2. Load the separate Landmark Predictor model ---
    lp_path = paths_cfg['inputs']['landmark_predictor_model']
    print(f"Loading Landmark Predictor from: {lp_path}")
    landmark_predictor = OcularLMGenerator().to(device)
    state_dict_lp = torch.load(lp_path, map_location=device)
    state_dict_lp = remove_module_prefix(state_dict_lp)
    landmark_predictor.load_state_dict(state_dict_lp)
    landmark_predictor.eval()

    # Return all models in a dictionary for easy access
    return {
        'netG': gan_models['G'],
        'netE': gan_models['E'],
        'landmark_encoder': gan_models['LE'],
        'landmark_predictor': landmark_predictor
    }

def load_image(file):
    """Safely loads an image file into a PIL Image object."""
    try:
        image = Image.open(file).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")
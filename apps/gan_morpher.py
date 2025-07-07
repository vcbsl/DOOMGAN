import torch
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from utils import create_landmark_heatmaps

def predict_landmarks_for_gan(image, landmark_predictor_model, device):
    """Predicts landmarks and formats them specifically for the GAN pipeline."""
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_transformed = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        landmarks = landmark_predictor_model(image_transformed).squeeze(0).cpu()
        landmarks = landmarks[:38] # Corresponds to 19 landmarks (x,y)
        landmarks = landmarks.view(-1, 2)
        # Normalize to [0, 1] range for heatmap generation
        landmarks[:, 0] /= 256.0
        landmarks[:, 1] /= 256.0
        landmarks = landmarks.flatten()
        
    return landmarks.unsqueeze(0) # Return with a batch dimension

def process_image_for_gan(image, config, device, models):
    """Processes a single image to get its latent vector (z) and landmark features (lf)."""
    image_tensor = Compose([
        Resize((config['data']['image_size'], config['data']['image_size'])),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])(image).unsqueeze(0).to(device)

    landmarks = predict_landmarks_for_gan(image, models['landmark_predictor'], device).to(device)
    heatmap = create_landmark_heatmaps(landmarks, image_size=config['data']['image_size']).to(device)
    
    with torch.no_grad():
        landmark_features = models['landmark_encoder'](landmarks)
        z = models['netE'](image_tensor, heatmap)
    
    return z, landmark_features

def morph_images_with_gan(image1, image2, config, device, models, alpha=0.5):
    """Generates a morphed image using the GAN with a given interpolation factor."""
    z1, lf1 = process_image_for_gan(image1, config, device, models)
    z2, lf2 = process_image_for_gan(image2, config, device, models)
    
    # Interpolate in both latent and landmark feature spaces
    z_morph = (1 - alpha) * z1 + alpha * z2
    lf_morph = (1 - alpha) * lf1 + alpha * lf2
    
    with torch.no_grad():
        morphed_image_tensor = models['netG'](z_morph, lf_morph)
    
    # Denormalize from [-1, 1] to [0, 1] for display
    morphed_image = (morphed_image_tensor * 0.5 + 0.5).clamp(0, 1)
    morphed_image_numpy = morphed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return morphed_image_numpy
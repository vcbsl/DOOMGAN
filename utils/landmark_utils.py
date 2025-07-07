import torch

def create_landmark_heatmaps(landmarks, image_size, sigma=2):
    """Generates Gaussian heatmaps for each landmark point."""
    b_size, n_landmarks = landmarks.size(0), landmarks.size(1) // 2
    heatmaps = torch.zeros(b_size, n_landmarks, image_size, image_size, device=landmarks.device)
    
    xx, yy = torch.meshgrid(
        torch.arange(image_size, device=landmarks.device, dtype=torch.float32),
        torch.arange(image_size, device=landmarks.device, dtype=torch.float32),
        indexing='ij'
    )
    
    for i in range(b_size):
        for j in range(n_landmarks):
            lx, ly = landmarks[i, j*2]*(image_size-1), landmarks[i, j*2+1]*(image_size-1)
            heatmaps[i, j] = torch.exp(-((xx - ly)**2 + (yy - lx)**2) / (2 * sigma**2))
            
    return heatmaps
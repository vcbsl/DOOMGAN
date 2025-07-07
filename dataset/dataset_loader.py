import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class LandmarkDataset(Dataset):
    """
    Custom PyTorch Dataset to load images and their corresponding landmark data.

    This dataset reads image paths and landmark coordinates from a JSON file.
    It assumes the keys in the JSON file are the full relative paths to the images
    from the project's root directory.
    """
    def __init__(self, landmark_json_path, image_size, transform=None, num_landmarks=19, **kwargs):
        """
        Initializes the dataset.

        Args:
            landmark_json_path (str): Path to the JSON file containing landmark data.
            image_size (int): The size to which images will be resized and landmarks normalized.
            transform (callable, optional): A torchvision transform to be applied to each image.
            num_landmarks (int, optional): The expected number of landmark points per image.
            **kwargs: Catches unused arguments from the config, like 'image_root'.
        """
        self.transform = transform
        self.image_size = image_size
        self.num_landmarks = num_landmarks

        # Load the landmark data from the JSON file
        with open(landmark_json_path, 'r') as f:
            self.landmarks_dict = json.load(f)

        self.image_paths = []
        self.landmark_data = []

        # Process paths directly from the JSON keys ---
        for image_path, landmarks_info in self.landmarks_dict.items():
            # Normalize path separators for cross-platform compatibility (e.g., Windows '\' vs. Linux '/')
            normalized_path = os.path.normpath(image_path)

            # Check if the file exists at the specified path
            if os.path.isfile(normalized_path):
                self.image_paths.append(normalized_path)
                self.landmark_data.append(landmarks_info)
            else:
                print(f"Warning: Image file not found and was skipped: {normalized_path}")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves a sample (image and landmarks) from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image tensor and the landmark tensor.
        """
        # The complete path is stored in self.image_paths ---
        full_image_path = self.image_paths[idx]
        
        try:
            image = Image.open(full_image_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find image at path: {full_image_path}")

        # Apply transformations if they exist
        if self.transform:
            image = self.transform(image)

        # Process the corresponding landmark data
        landmarks = self._process_landmarks(self.landmark_data[idx])

        return image, landmarks

    def _process_landmarks(self, landmarks_info):
        """
        Processes raw landmark data into a flattened, normalized tensor.
        Handles cases where landmarks might be missing by padding with zeros.
        """
        eye_landmarks = landmarks_info.get('Eye_landmarks', [])
        iris_landmarks = landmarks_info.get('Iris_landmarks', [])
        iris_center = landmarks_info.get('Iris_center', [])
        
        all_landmarks = eye_landmarks + iris_landmarks + iris_center

        # Pad or trim the landmarks list to ensure a consistent size
        if len(all_landmarks) < self.num_landmarks:
            missing_count = self.num_landmarks - len(all_landmarks)
            all_landmarks += [[0, 0]] * missing_count
        else:
            all_landmarks = all_landmarks[:self.num_landmarks]

        # Normalize landmarks to be in the range [0, 1] and flatten the list
        normalized_landmarks = [(x / self.image_size, y / self.image_size) for x, y in all_landmarks]
        landmarks_flat = [coord for point in normalized_landmarks for coord in point]
        
        return torch.tensor(landmarks_flat, dtype=torch.float32)

def custom_collate_fn(batch):
    """
    Custom collate function to stack images and landmarks from a batch.
    This is necessary if the default collate function has issues.
    """
    images, landmarks = zip(*batch)
    images = torch.stack(images, 0)
    landmarks = torch.stack(landmarks, 0)
    return images, landmarks

def get_dataloader(config):
    """
    Creates and returns a PyTorch DataLoader based on the provided configuration.

    Args:
        config (dict): A dictionary containing configuration parameters,
                       typically loaded from a YAML file.

    Returns:
        DataLoader: A configured DataLoader for the LandmarkDataset.
    """
    data_cfg = config['data']
    train_cfg = config['training']

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize(data_cfg['image_size']),
        transforms.CenterCrop(data_cfg['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Instantiate the dataset, passing all data-related configs.
    # The LandmarkDataset class will pick the arguments it needs.
    dataset = LandmarkDataset(transform=transform, **data_cfg)

    # Create and return the DataLoader
    return DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg.get('workers', 0), # Use .get for safety
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
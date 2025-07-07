import os
from pathlib import Path

# Define the directory structure
structure = {
    "config": ["__init__.py", "config.yaml"],
    "dataset": ["__init__.py", "dataset_loader.py"],
    "losses": ["__init__.py", "identity_loss.py", "msssim_loss.py", "perceptual_loss.py", "utils.py"],
    "models": ["__init__.py", "models.py", "test_models.py"],
    "utils": ["__init__.py", "dynamic_weight_adjuster.py", "landmark_utils.py"],
    "root": ["generate.py", "train.py", ".gitignore", "README.md"]
}

def create_structure(base_path="."):
    base_path = Path(base_path)
    
    for folder, files in structure.items():
        if folder != "root":
            dir_path = base_path / folder
            dir_path.mkdir(exist_ok=True)
            
            for file in files:
                file_path = dir_path / file
                file_path.touch()
        else:
            for file in files:
                file_path = base_path / file
                file_path.touch()
    
    print("Directory structure created successfully!")

if __name__ == "__main__":
    create_structure()
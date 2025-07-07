# All Required Model imports
from .models import (
    weights_init,
    SelfAttention,
    ResidualBlock,
    Encoder,
    Generator,
    Discriminator,
    LandmarkEncoder
)

# Import for the ArcFace model
try:
    from .ResNet_Model import ResNet50_ArcFace
except ImportError:
    ResNet50_ArcFace = None

# Import the LM Predictor model for App
try:
    from .landmark_predictor import OcularLMGenerator
except ImportError:
    OcularLMGenerator = None
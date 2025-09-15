import numpy as np
from typing import List
from .encoders import BaseEncoder
from fashion_clip.fashion_clip import FashionCLIP
from PIL import Image
from io import BytesIO
import requests
from utils.dataset_utils import download_image
class FashionCLIPEncoder(BaseEncoder):
    """
    Custom encoder for FashionCLIP model.
    """
    
    def __init__(self, device: str = "cpu", name: str = "fashionclip"):
        """
        Initialize FashionCLIP encoder.
        
        Args:
            model_path: Path to your FashionCLIP model
            device: Device to run on ('cpu', 'cuda')
            name: Name for caching
        """
        super().__init__(name)
        self.device = device
        
        # Initialize your FashionCLIP model here
        self.model = self._load_fashionclip_model()
        
    def _load_fashionclip_model(self):
        """
        Load your FashionCLIP model.
        Replace this with your actual model loading code.
        """
        print(f"Loading FashionCLIP ")
        
        fclip = FashionCLIP('fashion-clip')
        
        return fclip

    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using FashionCLIP.
        
        Args:
            texts: List of text descriptions
            
        Returns:
            Text embeddings array
        """
        embeddings = self.model.encode_text(texts, batch_size=32)
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def encode_image(self, image_paths: List[str]) -> np.ndarray:
        """
        Encode images using FashionCLIP.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Image embeddings array
        """
        # Load and preprocess images
        images = self._load_images(image_paths)
        
        embeddings = self.model.encode_images(images, batch_size=32)
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def _load_images(self, image_paths: List[str]):
        """
        Load and preprocess images.
        Replace with your actual image loading logic.
        """
        
        images = []
        for path in image_paths:
            try:
                if path.startswith('http'):
                    img = download_image(path)
                else:
                    img = Image.open(path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {path}: {e}")
                # Create dummy image or handle error
                images.append(Image.new('RGB', (224, 224)))
        
        return images
    
    def get_config_hash(self) -> str:
        """Include model path and device in hash for caching."""
        config_str = f"{self.name}_{self.model_path}_{self.device}"
        import hashlib
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

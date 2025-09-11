"""
Extensible encoder system for MTBE evaluations.
Supports custom models, HuggingFace models, and caching.
"""

import os
import hashlib
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseEncoder(ABC):
    """
    Abstract base class for text/image encoders.
    Users can implement custom encoders by inheriting from this class.
    """
    
    def __init__(self, name: str, cache_dir: str = "cache"):
        """
        Initialize the encoder.
        
        Args:
            name: Unique name for this encoder (used for caching)
            cache_dir: Directory to store cached embeddings
        """
        self.name = name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initialized {self.name} encoder")
    
    @abstractmethod
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text inputs to embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of text embeddings (n_texts, text_dim)
        """
        pass
    
    @abstractmethod
    def encode_image(self, image_paths: List[str]) -> np.ndarray:
        """
        Encode image inputs to embeddings.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Array of image embeddings (n_images, image_dim)
        """
        pass
    
    def get_config_hash(self) -> str:
        """
        Get a hash of the encoder configuration for caching.
        Override this method if your encoder has configurable parameters.
        
        Returns:
            Hash string representing the encoder configuration
        """
        config_str = f"{self.name}_{self.__class__.__name__}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_cache_path(self, dataset_name: str, data_type: str) -> str:
        """
        Get the cache file path for embeddings.
        
        Args:
            dataset_name: Name of the dataset
            data_type: Type of data ('catalog' or 'test')
            
        Returns:
            Path to cache file
        """
        config_hash = self.get_config_hash()
        cache_filename = f"{dataset_name}_{data_type}_{config_hash}.pkl"
        return os.path.join(self.cache_dir, self.name, cache_filename)
    
    def load_cached_embeddings(self, dataset_name: str, data_type: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load cached embeddings if they exist.
        
        Args:
            dataset_name: Name of the dataset
            data_type: Type of data ('catalog' or 'test')
            
        Returns:
            Dictionary with 'text_embeddings' and 'image_embeddings' or None
        """
        cache_path = self.get_cache_path(dataset_name, data_type)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Loaded cached embeddings from {cache_path}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
                return None
        
        return None
    
    def save_cached_embeddings(self, dataset_name: str, data_type: str, 
                              text_embeddings: np.ndarray, image_embeddings: np.ndarray):
        """
        Save embeddings to cache.
        
        Args:
            dataset_name: Name of the dataset
            data_type: Type of data ('catalog' or 'test')
            text_embeddings: Text embeddings array
            image_embeddings: Image embeddings array
        """
        cache_path = self.get_cache_path(dataset_name, data_type)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        cached_data = {
            'text_embeddings': text_embeddings,
            'image_embeddings': image_embeddings,
            'encoder_name': self.name,
            'config_hash': self.get_config_hash()
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            logger.info(f"Saved embeddings to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save embeddings to cache: {e}")
    
    def encode_dataset(self, texts: List[str], image_paths: List[str], 
                      dataset_name: str, data_type: str, 
                      use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode a complete dataset with caching support.
        
        Args:
            texts: List of text strings
            image_paths: List of image paths
            dataset_name: Name of the dataset
            data_type: Type of data ('catalog' or 'test')
            use_cache: Whether to use cached embeddings
            
        Returns:
            Tuple of (text_embeddings, image_embeddings)
        """
        # Try to load from cache first
        if use_cache:
            cached_data = self.load_cached_embeddings(dataset_name, data_type)
            if cached_data is not None:
                return cached_data['text_embeddings'], cached_data['image_embeddings']
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {dataset_name} {data_type} with {self.name}")
        text_embeddings = self.encode_text(texts)
        image_embeddings = self.encode_image(image_paths)
        
        # Save to cache
        if use_cache:
            self.save_cached_embeddings(dataset_name, data_type, text_embeddings, image_embeddings)
        
        return text_embeddings, image_embeddings


class MockEncoder(BaseEncoder):
    """
    Mock encoder for testing and demonstration.
    Generates random embeddings with configurable dimensions.
    """
    
    def __init__(self, text_dim: int = 128, image_dim: int = 128, name: str = "mock"):
        """
        Initialize mock encoder.
        
        Args:
            text_dim: Dimension of text embeddings
            image_dim: Dimension of image embeddings
            name: Name of the encoder
        """
        super().__init__(name)
        self.text_dim = text_dim
        self.image_dim = image_dim
        
        # Set random seed for reproducible embeddings
        np.random.seed(42)
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Generate random text embeddings."""
        embeddings = np.random.randn(len(texts), self.text_dim)
        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return embeddings
    
    def encode_image(self, image_paths: List[str]) -> np.ndarray:
        """Generate random image embeddings."""
        embeddings = np.random.randn(len(image_paths), self.image_dim)
        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return embeddings
    
    def get_config_hash(self) -> str:
        """Include dimensions in config hash."""
        config_str = f"{self.name}_{self.text_dim}_{self.image_dim}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class HuggingFaceEncoder(BaseEncoder):
    """
    Encoder using HuggingFace Sentence Transformers.
    Supports both text and image models.
    """
    
    def __init__(self, text_model_name: str, image_model_name: str = None, 
                 name: str = None, device: str = "cpu"):
        """
        Initialize HuggingFace encoder.
        
        Args:
            text_model_name: Name of the text model on HuggingFace
            image_model_name: Name of the image model (optional)
            name: Custom name for the encoder
            device: Device to run models on ('cpu', 'cuda', etc.)
        """
        if name is None:
            name = f"hf_{text_model_name.replace('/', '_')}"
        
        super().__init__(name)
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.device = device
        
        # Lazy loading of models
        self._text_model = None
        self._image_model = None
    
    @property
    def text_model(self):
        """Lazy load text model."""
        if self._text_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._text_model = SentenceTransformer(self.text_model_name, device=self.device)
                logger.info(f"Loaded text model: {self.text_model_name}")
            except ImportError:
                raise ImportError("sentence-transformers is required for HuggingFaceEncoder")
        return self._text_model
    
    @property
    def image_model(self):
        """Lazy load image model."""
        if self._image_model is None and self.image_model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self._image_model = SentenceTransformer(self.image_model_name, device=self.device)
                logger.info(f"Loaded image model: {self.image_model_name}")
            except ImportError:
                raise ImportError("sentence-transformers is required for HuggingFaceEncoder")
        return self._image_model
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts using HuggingFace model."""
        embeddings = self.text_model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def encode_image(self, image_paths: List[str]) -> np.ndarray:
        """Encode images using HuggingFace model."""
        if self.image_model_name is None:
            # Fallback to random embeddings if no image model specified
            logger.warning("No image model specified, using random embeddings")
            return np.random.randn(len(image_paths), 512)
        
        # Load images and encode
        from PIL import Image
        images = [Image.open(path) for path in image_paths]
        embeddings = self.image_model.encode(images, convert_to_numpy=True)
        return embeddings
    
    def get_config_hash(self) -> str:
        """Include model names in config hash."""
        config_str = f"{self.text_model_name}_{self.image_model_name}_{self.device}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class PrecomputedEncoder(BaseEncoder):
    """
    Encoder that uses pre-computed embeddings from CSV files.
    This is for backward compatibility with existing datasets.
    """
    
    def __init__(self, name: str = "precomputed"):
        super().__init__(name)
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Not used - embeddings are loaded from CSV."""
        raise NotImplementedError("PrecomputedEncoder loads embeddings from CSV files")
    
    def encode_image(self, image_paths: List[str]) -> np.ndarray:
        """Not used - embeddings are loaded from CSV."""
        raise NotImplementedError("PrecomputedEncoder loads embeddings from CSV files")
    
    def load_precomputed_embeddings(self, csv_path: str) -> tuple[List[str], List[str], np.ndarray, np.ndarray]:
        """
        Load pre-computed embeddings from CSV file.
        
        Args:
            csv_path: Path to CSV file with embeddings
            
        Returns:
            Tuple of (texts, image_paths, text_embeddings, image_embeddings)
        """
        import csv
        
        texts, image_paths = [], []
        text_embeddings, image_embeddings = [], []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row['text'])
                image_paths.append(row['image_path'])
                
                # Parse text embedding
                if 'text_embedding' in row and row['text_embedding']:
                    text_embedding_str = row['text_embedding'].strip('[]')
                    text_embedding = [float(x.strip()) for x in text_embedding_str.split(',')]
                    text_embeddings.append(text_embedding)
                else:
                    text_embeddings.append([])
                
                # Parse image embedding
                if 'image_embedding' in row and row['image_embedding']:
                    image_embedding_str = row['image_embedding'].strip('[]')
                    image_embedding = [float(x.strip()) for x in image_embedding_str.split(',')]
                    image_embeddings.append(image_embedding)
                else:
                    image_embeddings.append([])
        
        # Convert to numpy arrays
        text_embeddings_array = np.array(text_embeddings) if text_embeddings and text_embeddings[0] else np.array([])
        image_embeddings_array = np.array(image_embeddings) if image_embeddings and image_embeddings[0] else np.array([])
        
        return texts, image_paths, text_embeddings_array, image_embeddings_array


def get_encoder(encoder_config: Union[str, Dict[str, Any]]) -> BaseEncoder:
    """
    Factory function to create encoders from configuration.
    
    Args:
        encoder_config: Either a string name or configuration dictionary
        
    Returns:
        Initialized encoder instance
    """
    if isinstance(encoder_config, str):
        # Simple string configuration
        if encoder_config == "mock":
            return MockEncoder()
        elif encoder_config == "precomputed":
            return PrecomputedEncoder()
        else:
            # Assume it's a HuggingFace model name
            return HuggingFaceEncoder(text_model_name=encoder_config)
    
    elif isinstance(encoder_config, dict):
        # Dictionary configuration
        encoder_type = encoder_config.get("type", "mock")
        
        if encoder_type == "mock":
            return MockEncoder(
                text_dim=encoder_config.get("text_dim", 128),
                image_dim=encoder_config.get("image_dim", 128),
                name=encoder_config.get("name", "mock")
            )
        
        elif encoder_type == "huggingface":
            return HuggingFaceEncoder(
                text_model_name=encoder_config["text_model"],
                image_model_name=encoder_config.get("image_model"),
                name=encoder_config.get("name"),
                device=encoder_config.get("device", "cpu")
            )
        
        elif encoder_type == "precomputed":
            return PrecomputedEncoder(name=encoder_config.get("name", "precomputed"))
        
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    else:
        raise ValueError(f"Invalid encoder configuration: {encoder_config}")

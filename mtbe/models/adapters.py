"""
Extensible adapter system for transforming embeddings in MTBE evaluations.
Supports custom adapters and caching.
"""

import os
import hashlib
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """
    Abstract base class for embedding adapters.
    Users can implement custom adapters by inheriting from this class.
    """
    
    def __init__(self, name: str, cache_dir: str = "cache"):
        """
        Initialize the adapter.
        
        Args:
            name: Unique name for this adapter (used for caching)
            cache_dir: Directory to store cached transformed embeddings
        """
        self.name = name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initialized {self.name} adapter")
    
    @abstractmethod
    def transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using the adapter strategy.
        
        Args:
            embeddings: Input embeddings array (n_items, embedding_dim)
            
        Returns:
            Transformed embeddings array (n_items, output_dim)
        """
        pass
    
    def fit(self, embeddings: np.ndarray) -> 'BaseAdapter':
        """
        Fit the adapter to training embeddings (optional).
        Override this method if your adapter needs training.
        
        Args:
            embeddings: Training embeddings
            
        Returns:
            Self for method chaining
        """
        return self
    
    def get_config_hash(self) -> str:
        """
        Get a hash of the adapter configuration for caching.
        Override this method if your adapter has configurable parameters.
        
        Returns:
            Hash string representing the adapter configuration
        """
        config_str = f"{self.name}_{self.__class__.__name__}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_cache_path(self, encoder_name: str, dataset_name: str, data_type: str) -> str:
        """
        Get the cache file path for transformed embeddings.
        
        Args:
            encoder_name: Name of the encoder used
            dataset_name: Name of the dataset
            data_type: Type of data ('catalog' or 'test')
            
        Returns:
            Path to cache file
        """
        config_hash = self.get_config_hash()
        cache_filename = f"{encoder_name}_{dataset_name}_{data_type}_{config_hash}.pkl"
        return os.path.join(self.cache_dir, self.name, cache_filename)
    
    def load_cached_embeddings(self, encoder_name: str, dataset_name: str, 
                              data_type: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load cached transformed embeddings if they exist.
        
        Args:
            encoder_name: Name of the encoder used
            dataset_name: Name of the dataset
            data_type: Type of data ('catalog' or 'test')
            
        Returns:
            Dictionary with 'text_embeddings' and 'image_embeddings' or None
        """
        cache_path = self.get_cache_path(encoder_name, dataset_name, data_type)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Loaded cached transformed embeddings from {cache_path}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cached transformed embeddings: {e}")
                return None
        
        return None
    
    def save_cached_embeddings(self, encoder_name: str, dataset_name: str, data_type: str,
                              text_embeddings: np.ndarray, image_embeddings: np.ndarray):
        """
        Save transformed embeddings to cache.
        
        Args:
            encoder_name: Name of the encoder used
            dataset_name: Name of the dataset
            data_type: Type of data ('catalog' or 'test')
            text_embeddings: Transformed text embeddings array
            image_embeddings: Transformed image embeddings array
        """
        cache_path = self.get_cache_path(encoder_name, dataset_name, data_type)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        cached_data = {
            'text_embeddings': text_embeddings,
            'image_embeddings': image_embeddings,
            'adapter_name': self.name,
            'config_hash': self.get_config_hash()
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            logger.info(f"Saved transformed embeddings to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save transformed embeddings to cache: {e}")
    
    def transform_dataset(self, text_embeddings: np.ndarray, image_embeddings: np.ndarray,
                         encoder_name: str, dataset_name: str, data_type: str,
                         use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform a complete dataset with caching support.
        
        Args:
            text_embeddings: Input text embeddings
            image_embeddings: Input image embeddings
            encoder_name: Name of the encoder used
            dataset_name: Name of the dataset
            data_type: Type of data ('catalog' or 'test')
            use_cache: Whether to use cached embeddings
            
        Returns:
            Tuple of (transformed_text_embeddings, transformed_image_embeddings)
        """
        # Try to load from cache first
        if use_cache:
            cached_data = self.load_cached_embeddings(encoder_name, dataset_name, data_type)
            if cached_data is not None:
                return cached_data['text_embeddings'], cached_data['image_embeddings']
        
        # Transform embeddings
        logger.info(f"Transforming embeddings for {dataset_name} {data_type} with {self.name}")
        transformed_text = self.transform_embeddings(text_embeddings)
        transformed_image = self.transform_embeddings(image_embeddings)
        
        # Save to cache
        if use_cache:
            self.save_cached_embeddings(encoder_name, dataset_name, data_type, 
                                       transformed_text, transformed_image)
        
        return transformed_text, transformed_image


class IdentityAdapter(BaseAdapter):
    """
    Identity adapter that returns embeddings unchanged.
    Useful as a baseline or when no transformation is needed.
    """
    
    def __init__(self, name: str = "identity"):
        super().__init__(name)
    
    def transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Return embeddings unchanged."""
        return embeddings.copy()


class NormalizationAdapter(BaseAdapter):
    """
    Adapter that normalizes embeddings to unit length.
    """
    
    def __init__(self, name: str = "normalize"):
        super().__init__(name)
    
    def transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms


class PCAAdapter(BaseAdapter):
    """
    Adapter that applies Principal Component Analysis (PCA) dimensionality reduction.
    """
    
    def __init__(self, n_components: int, name: str = None):
        """
        Initialize PCA adapter.
        
        Args:
            n_components: Number of principal components to keep
            name: Custom name for the adapter
        """
        if name is None:
            name = f"pca_{n_components}"
        
        super().__init__(name)
        self.n_components = n_components
        self.pca_model = None
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray) -> 'PCAAdapter':
        """
        Fit PCA model to training embeddings.
        
        Args:
            embeddings: Training embeddings
            
        Returns:
            Self for method chaining
        """
        try:
            from sklearn.decomposition import PCA
            
            self.pca_model = PCA(n_components=self.n_components)
            self.pca_model.fit(embeddings)
            self.is_fitted = True
            
            logger.info(f"Fitted PCA with {self.n_components} components")
            logger.info(f"Explained variance ratio: {self.pca_model.explained_variance_ratio_[:5]}")
            
        except ImportError:
            raise ImportError("scikit-learn is required for PCAAdapter")
        
        return self
    
    def transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA transformation."""
        if not self.is_fitted:
            raise ValueError("PCA adapter must be fitted before transformation")
        
        return self.pca_model.transform(embeddings)
    
    def get_config_hash(self) -> str:
        """Include n_components in config hash."""
        config_str = f"{self.name}_{self.n_components}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class RotationAdapter(BaseAdapter):
    """
    Adapter that applies a random rotation matrix to embeddings.
    Useful for testing robustness to orthogonal transformations.
    """
    
    def __init__(self, seed: int = 42, name: str = None):
        """
        Initialize rotation adapter.
        
        Args:
            seed: Random seed for reproducible rotations
            name: Custom name for the adapter
        """
        if name is None:
            name = f"rotation_{seed}"
        
        super().__init__(name)
        self.seed = seed
        self.rotation_matrix = None
        self.embedding_dim = None
    
    def _generate_rotation_matrix(self, dim: int) -> np.ndarray:
        """Generate a random orthogonal rotation matrix."""
        np.random.seed(self.seed)
        # Generate random matrix and apply QR decomposition
        random_matrix = np.random.randn(dim, dim)
        q, r = np.linalg.qr(random_matrix)
        # Ensure proper rotation (det = 1)
        q = q * np.linalg.det(q)
        return q
    
    def transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply rotation transformation."""
        if embeddings.size == 0:
            return embeddings
        
        dim = embeddings.shape[1]
        
        # Generate rotation matrix if needed
        if self.rotation_matrix is None or self.embedding_dim != dim:
            self.rotation_matrix = self._generate_rotation_matrix(dim)
            self.embedding_dim = dim
            logger.info(f"Generated {dim}x{dim} rotation matrix")
        
        return embeddings @ self.rotation_matrix
    
    def get_config_hash(self) -> str:
        """Include seed in config hash."""
        config_str = f"{self.name}_{self.seed}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ScalingAdapter(BaseAdapter):
    """
    Adapter that scales embeddings by a constant factor.
    """
    
    def __init__(self, scale_factor: float = 1.0, name: str = None):
        """
        Initialize scaling adapter.
        
        Args:
            scale_factor: Factor to scale embeddings by
            name: Custom name for the adapter
        """
        if name is None:
            name = f"scale_{scale_factor}"
        
        super().__init__(name)
        self.scale_factor = scale_factor
    
    def transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Scale embeddings by constant factor."""
        return embeddings * self.scale_factor
    
    def get_config_hash(self) -> str:
        """Include scale factor in config hash."""
        config_str = f"{self.name}_{self.scale_factor}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class CompositeAdapter(BaseAdapter):
    """
    Adapter that applies multiple adapters in sequence.
    """
    
    def __init__(self, adapters: List[BaseAdapter], name: str = None):
        """
        Initialize composite adapter.
        
        Args:
            adapters: List of adapters to apply in sequence
            name: Custom name for the adapter
        """
        if name is None:
            adapter_names = "_".join([adapter.name for adapter in adapters])
            name = f"composite_{adapter_names}"
        
        super().__init__(name)
        self.adapters = adapters
    
    def fit(self, embeddings: np.ndarray) -> 'CompositeAdapter':
        """
        Fit all adapters in sequence.
        
        Args:
            embeddings: Training embeddings
            
        Returns:
            Self for method chaining
        """
        current_embeddings = embeddings
        
        for adapter in self.adapters:
            adapter.fit(current_embeddings)
            current_embeddings = adapter.transform_embeddings(current_embeddings)
        
        return self
    
    def transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply all adapters in sequence."""
        current_embeddings = embeddings
        
        for adapter in self.adapters:
            current_embeddings = adapter.transform_embeddings(current_embeddings)
        
        return current_embeddings
    
    def get_config_hash(self) -> str:
        """Include all adapter hashes in config hash."""
        adapter_hashes = [adapter.get_config_hash() for adapter in self.adapters]
        config_str = f"{self.name}_{'_'.join(adapter_hashes)}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_adapter(adapter_config: Union[str, Dict[str, Any], List]) -> BaseAdapter:
    """
    Factory function to create adapters from configuration.
    
    Args:
        adapter_config: String name, configuration dictionary, or list of configs
        
    Returns:
        Initialized adapter instance
    """
    if isinstance(adapter_config, str):
        # Simple string configuration
        if adapter_config == "identity":
            return IdentityAdapter()
        elif adapter_config == "normalize":
            return NormalizationAdapter()
        else:
            raise ValueError(f"Unknown adapter type: {adapter_config}")
    
    elif isinstance(adapter_config, dict):
        # Dictionary configuration
        adapter_type = adapter_config.get("type", "identity")
        
        if adapter_type == "identity":
            return IdentityAdapter(name=adapter_config.get("name", "identity"))
        
        elif adapter_type == "normalize":
            return NormalizationAdapter(name=adapter_config.get("name", "normalize"))
        
        elif adapter_type == "pca":
            return PCAAdapter(
                n_components=adapter_config["n_components"],
                name=adapter_config.get("name")
            )
        
        elif adapter_type == "rotation":
            return RotationAdapter(
                seed=adapter_config.get("seed", 42),
                name=adapter_config.get("name")
            )
        
        elif adapter_type == "scaling":
            return ScalingAdapter(
                scale_factor=adapter_config.get("scale_factor", 1.0),
                name=adapter_config.get("name")
            )
        
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    elif isinstance(adapter_config, list):
        # List of adapter configurations (composite adapter)
        adapters = [get_adapter(config) for config in adapter_config]
        return CompositeAdapter(adapters)
    
    else:
        raise ValueError(f"Invalid adapter configuration: {adapter_config}")

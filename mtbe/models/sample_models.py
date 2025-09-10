from __future__ import annotations

import numpy as np
from datetime import date
from typing import Any

from mtbe.encoder_interface import Encoder, MultimodalEncoder
from mtbe.model_meta import ModelMeta


class SampleTextEncoder(Encoder):
    """Sample text encoder for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "sample-text-encoder"
        self.embedding_dim = 384
        
        # Model metadata
        self.mteb_model_meta = ModelMeta(
            name="sample-text-encoder",
            revision="v1.0",
            release_date=date(2024, 1, 1),
            languages=["eng"],
            modalities=["text"],
            framework="custom",
            model_type="sentence-transformer",
            embedding_size=self.embedding_dim,
            max_tokens=512,
            license="MIT",
            similarity_fn_name="cosine",
        )
    
    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode sentences to embeddings.
        
        This is a dummy implementation that generates random embeddings.
        In a real implementation, this would use an actual model.
        """
        # Generate random embeddings (for demo purposes)
        np.random.seed(42)  # For reproducible results
        embeddings = []
        
        for sentence in sentences:
            # Create a simple hash-based embedding for consistency
            sentence_hash = hash(sentence) % 1000000
            np.random.seed(sentence_hash)
            embedding = np.random.normal(0, 1, self.embedding_dim)
            # Normalize to unit length
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)


class SampleMultimodalEncoder(MultimodalEncoder):
    """Sample multimodal encoder for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "sample-multimodal-encoder"
        self.text_embedding_dim = 384
        self.image_embedding_dim = 384
        self.multimodal_embedding_dim = 768
        
        # Model metadata
        self.mteb_model_meta = ModelMeta(
            name="sample-multimodal-encoder",
            revision="v1.0",
            release_date=date(2024, 1, 1),
            languages=["eng"],
            modalities=["text", "image"],
            framework="custom",
            model_type="multimodal-transformer",
            embedding_size=self.multimodal_embedding_dim,
            max_tokens=512,
            license="MIT",
            similarity_fn_name="cosine",
        )
    
    def encode_text(
        self,
        texts: list[str],
        *,
        task_name: str,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode text inputs."""
        np.random.seed(42)
        embeddings = []
        
        for text in texts:
            # Create a simple hash-based embedding for consistency
            text_hash = hash(text) % 1000000
            np.random.seed(text_hash)
            embedding = np.random.normal(0, 1, self.text_embedding_dim)
            # Normalize to unit length
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode_image(
        self,
        images: list[str],
        *,
        task_name: str,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode image inputs.
        
        Args:
            images: List of image paths or PIL Images
        """
        np.random.seed(42)
        embeddings = []
        
        for image_path in images:
            # Create a simple hash-based embedding for consistency
            # In a real implementation, this would load and process the image
            image_hash = hash(str(image_path)) % 1000000
            np.random.seed(image_hash)
            embedding = np.random.normal(0, 1, self.image_embedding_dim)
            # Normalize to unit length
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode_multimodal(
        self,
        texts: list[str],
        images: list[str],
        *,
        task_name: str,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode multimodal inputs (text + image)."""
        if len(texts) != len(images):
            raise ValueError("Number of texts and images must be equal")
        
        text_embeddings = self.encode_text(texts, task_name=task_name, **kwargs)
        image_embeddings = self.encode_image(images, task_name=task_name, **kwargs)
        
        # Simple concatenation (in a real model, this would be more sophisticated)
        multimodal_embeddings = np.concatenate([text_embeddings, image_embeddings], axis=1)
        
        # Normalize the combined embeddings
        norms = np.linalg.norm(multimodal_embeddings, axis=1, keepdims=True)
        multimodal_embeddings = multimodal_embeddings / norms
        
        return multimodal_embeddings.astype(np.float32)


class SampleCLIPModel(MultimodalEncoder):
    """Sample CLIP-like model for more realistic testing."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "sample-clip-model"
        self.embedding_dim = 512
        
        # Model metadata
        self.mteb_model_meta = ModelMeta(
            name="sample-clip-model",
            revision="v1.0",
            release_date=date(2024, 1, 1),
            languages=["eng"],
            modalities=["text", "image"],
            framework="custom",
            model_type="clip",
            embedding_size=self.embedding_dim,
            max_tokens=77,
            license="MIT",
            similarity_fn_name="cosine",
        )
    
    def encode_text(
        self,
        texts: list[str],
        *,
        task_name: str,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode text using CLIP-like text encoder."""
        embeddings = []
        
        for text in texts:
            # Simulate CLIP text encoding with more realistic patterns
            text_hash = hash(text) % 1000000
            np.random.seed(text_hash)
            
            # Create embedding with some structure
            embedding = np.random.normal(0, 0.1, self.embedding_dim)
            
            # Add some semantic structure based on text content
            if "cat" in text.lower() or "dog" in text.lower():
                embedding[:50] += 0.5  # Animal features
            if "car" in text.lower() or "vehicle" in text.lower():
                embedding[50:100] += 0.5  # Vehicle features
            if "red" in text.lower() or "blue" in text.lower():
                embedding[100:150] += 0.3  # Color features
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode_image(
        self,
        images: list[str],
        *,
        task_name: str,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode images using CLIP-like image encoder."""
        embeddings = []
        
        for image_path in images:
            # Simulate CLIP image encoding
            image_hash = hash(str(image_path)) % 1000000
            np.random.seed(image_hash)
            
            # Create embedding with structure based on image path
            embedding = np.random.normal(0, 0.1, self.embedding_dim)
            
            # Add semantic structure based on image filename
            filename = str(image_path).lower()
            if "cat" in filename or "dog" in filename:
                embedding[:50] += 0.5  # Animal features
            if "car" in filename or "vehicle" in filename:
                embedding[50:100] += 0.5  # Vehicle features
            if "red" in filename or "blue" in filename:
                embedding[100:150] += 0.3  # Color features
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)

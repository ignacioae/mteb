import os
import time
import asyncio
import hashlib
import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import base64
import io
from PIL import Image

try:
    from google.cloud import aiplatform
    from google.auth import default
    from google.auth.exceptions import DefaultCredentialsError
    from vertexai.preview.vision_models import MultiModalEmbeddingModel, Image as VertexImage
    from vertexai.language_models import TextEmbeddingModel
    import vertexai
    GOOGLE_AVAILABLE = True
except ImportError as e:
    GOOGLE_AVAILABLE = False
    IMPORT_ERROR = str(e)

from .encoders import BaseEncoder

logger = logging.getLogger(__name__)


@dataclass
class GoogleAPIConfig:
    """Configuration for Google API encoder."""
    project_id: str
    location: str = "us-central1"
    text_model: str = "text-embedding-004"
    image_model: str = "multimodalembedding@001"
    credentials_path: Optional[str] = None
    rate_limit_requests_per_minute: int = 60
    batch_size: int = 5
    max_retries: int = 3
    timeout: int = 30
    text_embedding_dimension: int = 768
    image_embedding_dimension: int = 1408


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
    
    async def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()


class GoogleAPIEncoder(BaseEncoder):
    """
    Encoder that uses Google Cloud Vertex AI APIs for generating embeddings.
    
    Supports both text and image embeddings using Google's latest models:
    - Text: text-embedding-004, text-multilingual-embedding-002
    - Images: multimodalembedding@001
    
    Features:
    - Rate limiting and batch processing
    - Automatic retries with exponential backoff
    - Comprehensive error handling
    - Caching support
    - Async processing for better performance
    
    Example usage:
        # Simple configuration
        encoder = GoogleAPIEncoder(project_id="your-project-id")
        
        # Advanced configuration
        encoder = GoogleAPIEncoder(
            project_id="your-project-id",
            location="us-central1",
            text_model="text-embedding-004",
            image_model="multimodalembedding@001",
            rate_limit_requests_per_minute=60,
            batch_size=5
        )
    """
    
    def __init__(self, 
                 project_id: str,
                 location: str = "us-central1",
                 text_model: str = "text-embedding-004",
                 image_model: str = "multimodalembedding@001",
                 credentials_path: Optional[str] = None,
                 rate_limit_requests_per_minute: int = 60,
                 batch_size: int = 5,
                 max_retries: int = 3,
                 timeout: int = 30,
                 name: str = "google_api",
                 cache_dir: str = "cache"):
        """
        Initialize Google API encoder.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region (default: us-central1)
            text_model: Text embedding model name
            image_model: Image embedding model name
            credentials_path: Path to service account JSON file (optional)
            rate_limit_requests_per_minute: API rate limit
            batch_size: Number of items to process in each batch
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            name: Encoder name for caching
            cache_dir: Directory for cached embeddings
        """
        super().__init__(name, cache_dir)
        
        # Check if Google Cloud libraries are available
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                f"Google Cloud libraries not available: {IMPORT_ERROR}\n"
                "Please install with: pip install google-cloud-aiplatform vertexai"
            )
        
        # Store configuration
        self.config = GoogleAPIConfig(
            project_id=project_id,
            location=location,
            text_model=text_model,
            image_model=image_model,
            credentials_path=credentials_path,
            rate_limit_requests_per_minute=rate_limit_requests_per_minute,
            batch_size=batch_size,
            max_retries=max_retries,
            timeout=timeout
        )
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(rate_limit_requests_per_minute)
        
        # Initialize Google Cloud
        self._initialize_google_cloud()
        
        # Load models
        self._load_models()
        
        logger.info(f"Initialized Google API encoder with project: {project_id}")
    
    def _initialize_google_cloud(self):
        """Initialize Google Cloud authentication and project."""
        try:
            # Set credentials if provided
            if self.config.credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.credentials_path
            
            # Initialize Vertex AI
            vertexai.init(
                project=self.config.project_id,
                location=self.config.location
            )
            
            # Test authentication
            credentials, project = default()
            logger.info(f"Successfully authenticated with Google Cloud project: {project}")
            
        except DefaultCredentialsError as e:
            raise ValueError(
                f"Google Cloud authentication failed: {e}\n"
                "Please set up authentication using one of:\n"
                "1. Service account key file (credentials_path parameter)\n"
                "2. GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
                "3. gcloud auth application-default login"
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Google Cloud: {e}")
    
    def _load_models(self):
        """Load Google Cloud embedding models."""
        try:
            # Load text embedding model
            self.text_model = TextEmbeddingModel.from_pretrained(self.config.text_model)
            logger.info(f"Loaded text model: {self.config.text_model}")
            
            # Load multimodal embedding model for images
            self.image_model = MultiModalEmbeddingModel.from_pretrained(self.config.image_model)
            logger.info(f"Loaded image model: {self.config.image_model}")
            
        except Exception as e:
            raise ValueError(f"Failed to load Google Cloud models: {e}")
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                await self.rate_limiter.wait_if_needed()
                return await func(*args, **kwargs)
            
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise e
                
                wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
    
    async def _encode_text_batch_async(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts asynchronously."""
        try:
            # Get embeddings from Google API
            embeddings = self.text_model.get_embeddings(texts)
            
            # Convert to numpy array
            embedding_vectors = []
            for embedding in embeddings:
                embedding_vectors.append(embedding.values)
            
            return np.array(embedding_vectors)
            
        except Exception as e:
            logger.error(f"Failed to encode text batch: {e}")
            raise
    
    async def _encode_image_batch_async(self, image_paths: List[str]) -> np.ndarray:
        """Encode a batch of images asynchronously."""
        try:
            # Load images
            vertex_images = []
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    # Create a dummy embedding for missing images
                    vertex_images.append(None)
                    continue
                
                # Load image using Vertex AI format
                vertex_image = VertexImage.load_from_file(image_path)
                vertex_images.append(vertex_image)
            
            # Get embeddings from Google API
            embeddings = []
            for i, vertex_image in enumerate(vertex_images):
                if vertex_image is None:
                    # Create zero embedding for missing images
                    zero_embedding = np.zeros(self.config.image_embedding_dimension)
                    embeddings.append(zero_embedding)
                else:
                    embedding = self.image_model.get_embeddings(
                        image=vertex_image,
                        contextual_text=None
                    )
                    embeddings.append(embedding.image_embedding)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to encode image batch: {e}")
            raise
    
    def _process_in_batches(self, items: List, encode_func) -> np.ndarray:
        """Process items in batches using async processing."""
        if not items:
            return np.array([])
        
        # Split into batches
        batches = [
            items[i:i + self.config.batch_size] 
            for i in range(0, len(items), self.config.batch_size)
        ]
        
        # Process batches
        all_embeddings = []
        
        async def process_all_batches():
            tasks = []
            for batch in batches:
                task = self._retry_with_backoff(encode_func, batch)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        # Run async processing
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        batch_results = loop.run_until_complete(process_all_batches())
        
        # Combine results
        for batch_embeddings in batch_results:
            if len(batch_embeddings) > 0:
                all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings) if all_embeddings else np.array([])
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text inputs to embeddings using Google Cloud API.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of text embeddings (n_texts, text_dim)
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} texts with Google API")
        
        try:
            embeddings = self._process_in_batches(texts, self._encode_text_batch_async)
            logger.info(f"Successfully encoded {len(embeddings)} text embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    def encode_image(self, image_paths: List[str]) -> np.ndarray:
        """
        Encode image inputs to embeddings using Google Cloud API.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Array of image embeddings (n_images, image_dim)
        """
        if not image_paths:
            return np.array([])
        
        logger.info(f"Encoding {len(image_paths)} images with Google API")
        
        try:
            embeddings = self._process_in_batches(image_paths, self._encode_image_batch_async)
            logger.info(f"Successfully encoded {len(embeddings)} image embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode images: {e}")
            raise
    
    def get_config_hash(self) -> str:
        """Get configuration hash for caching."""
        config_str = (
            f"{self.name}_{self.config.project_id}_{self.config.text_model}_"
            f"{self.config.image_model}_{self.config.location}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models."""
        return {
            "text_model": self.config.text_model,
            "image_model": self.config.image_model,
            "project_id": self.config.project_id,
            "location": self.config.location,
            "text_embedding_dimension": self.config.text_embedding_dimension,
            "image_embedding_dimension": self.config.image_embedding_dimension,
            "rate_limit": self.config.rate_limit_requests_per_minute,
            "batch_size": self.config.batch_size
        }


# Convenience function for easy instantiation
def create_google_api_encoder(project_id: str, **kwargs) -> GoogleAPIEncoder:
    """
    Create a Google API encoder with simplified configuration.
    
    Args:
        project_id: Google Cloud project ID
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured GoogleAPIEncoder instance
        
    Example:
        encoder = create_google_api_encoder(
            project_id="my-project",
            location="us-central1",
            rate_limit_requests_per_minute=30
        )
    """
    return GoogleAPIEncoder(project_id=project_id, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python google_api_encoder.py <project_id>")
        sys.exit(1)
    
    project_id = sys.argv[1]
    
    # Create encoder
    encoder = GoogleAPIEncoder(
        project_id=project_id,
        rate_limit_requests_per_minute=10,  # Conservative for testing
        batch_size=2
    )
    
    # Test text encoding
    test_texts = [
        "This is a test sentence for embedding.",
        "Another example text for testing the API."
    ]
    
    print("Testing text encoding...")
    text_embeddings = encoder.encode_text(test_texts)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Test image encoding (if images are available)
    test_images = ["test_image1.jpg", "test_image2.jpg"]  # Replace with actual paths
    
    print("Testing image encoding...")
    try:
        image_embeddings = encoder.encode_image(test_images)
        print(f"Image embeddings shape: {image_embeddings.shape}")
    except Exception as e:
        print(f"Image encoding test failed (expected if test images don't exist): {e}")
    
    # Print model info
    print("\nModel information:")
    info = encoder.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

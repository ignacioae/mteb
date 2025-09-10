"""Utility functions for the Cline Version framework."""

import os
import hashlib
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Union
import logging

logger = logging.getLogger(__name__)


def is_url(path: str) -> bool:
    """Check if a path is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def download_image_from_url(url: str, cache_dir: str = None) -> str:
    """Download an image from URL and return local path.
    
    Args:
        url: URL of the image to download
        cache_dir: Directory to cache downloaded images. If None, uses temp directory.
        
    Returns:
        Local path to the downloaded image
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required for downloading images from URLs. Install with: pip install requests")
    
    # Create cache directory
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "cline_version_image_cache")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create filename from URL hash
    url_hash = hashlib.md5(url.encode()).hexdigest()
    
    # Try to get file extension from URL
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.split('.')
    if len(path_parts) > 1:
        extension = path_parts[-1].lower()
        if extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
            filename = f"{url_hash}.{extension}"
        else:
            filename = f"{url_hash}.jpg"  # Default to jpg
    else:
        filename = f"{url_hash}.jpg"
    
    local_path = os.path.join(cache_dir, filename)
    
    # Check if already downloaded
    if os.path.exists(local_path):
        logger.debug(f"Using cached image: {local_path}")
        return local_path
    
    # Download the image
    try:
        logger.info(f"Downloading image from: {url}")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.debug(f"Image downloaded to: {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise


def resolve_image_path(image_path: Union[str, Path], cache_dir: str = None) -> str:
    """Resolve image path, downloading from URL if necessary.
    
    Args:
        image_path: Local path or URL to image
        cache_dir: Directory to cache downloaded images
        
    Returns:
        Local path to the image
    """
    image_path_str = str(image_path)
    
    if is_url(image_path_str):
        return download_image_from_url(image_path_str, cache_dir)
    else:
        # Local path - return as is
        return image_path_str


def batch_resolve_image_paths(image_paths: list[Union[str, Path]], cache_dir: str = None) -> list[str]:
    """Resolve multiple image paths, downloading from URLs if necessary.
    
    Args:
        image_paths: List of local paths or URLs to images
        cache_dir: Directory to cache downloaded images
        
    Returns:
        List of local paths to the images
    """
    resolved_paths = []
    
    for image_path in image_paths:
        try:
            resolved_path = resolve_image_path(image_path, cache_dir)
            resolved_paths.append(resolved_path)
        except Exception as e:
            logger.error(f"Failed to resolve image path {image_path}: {e}")
            # Use original path as fallback
            resolved_paths.append(str(image_path))
    
    return resolved_paths

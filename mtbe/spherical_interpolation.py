"""
Spherical interpolation utilities for multimodal embeddings.
"""

import numpy as np
from typing import Union, List
import torch


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding to unit length."""
    norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
    return embedding / (norm + 1e-8)


def spherical_interpolation(
    text_embedding: np.ndarray,
    image_embedding: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Perform spherical linear interpolation (SLERP) between text and image embeddings.
    
    Args:
        text_embedding: Text embedding vector(s)
        image_embedding: Image embedding vector(s)
        alpha: Interpolation weight (0.0 = text only, 1.0 = image only)
        
    Returns:
        Interpolated embedding
    """
    if alpha == 0.0:
        return text_embedding
    elif alpha == 1.0:
        return image_embedding
    
    # Normalize embeddings
    text_norm = normalize_embedding(text_embedding)
    image_norm = normalize_embedding(image_embedding)
    
    # Calculate angle between embeddings
    dot_product = np.sum(text_norm * image_norm, axis=-1, keepdims=True)
    # Clamp to avoid numerical issues
    dot_product = np.clip(dot_product, -1.0, 1.0)
    omega = np.arccos(np.abs(dot_product))
    
    # Handle case where embeddings are nearly parallel
    sin_omega = np.sin(omega)
    parallel_mask = sin_omega < 1e-6
    
    if np.any(parallel_mask):
        # Use linear interpolation for nearly parallel vectors
        result = (1 - alpha) * text_norm + alpha * image_norm
        result = normalize_embedding(result)
        return result
    
    # Spherical interpolation
    factor1 = np.sin((1 - alpha) * omega) / sin_omega
    factor2 = np.sin(alpha * omega) / sin_omega
    
    interpolated = factor1 * text_norm + factor2 * image_norm
    return normalize_embedding(interpolated)


def batch_spherical_interpolation(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Perform spherical interpolation on batches of embeddings.
    
    Args:
        text_embeddings: Array of text embeddings (N, D)
        image_embeddings: Array of image embeddings (N, D)
        alpha: Interpolation weight
        
    Returns:
        Array of interpolated embeddings (N, D)
    """
    assert text_embeddings.shape == image_embeddings.shape, \
        "Text and image embeddings must have the same shape"
    
    if len(text_embeddings.shape) == 1:
        return spherical_interpolation(text_embeddings, image_embeddings, alpha)
    
    results = []
    for i in range(text_embeddings.shape[0]):
        interpolated = spherical_interpolation(
            text_embeddings[i], 
            image_embeddings[i], 
            alpha
        )
        results.append(interpolated)
    
    return np.array(results)


def interpolate_embeddings_with_alpha_sweep(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    alpha_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
) -> dict:
    """
    Generate interpolated embeddings for multiple alpha values.
    
    Args:
        text_embeddings: Array of text embeddings
        image_embeddings: Array of image embeddings
        alpha_values: List of interpolation weights to test
        
    Returns:
        Dictionary mapping alpha values to interpolated embeddings
    """
    results = {}
    
    for alpha in alpha_values:
        interpolated = batch_spherical_interpolation(
            text_embeddings, 
            image_embeddings, 
            alpha
        )
        results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "embeddings": interpolated,
            "modality_focus": get_modality_focus(alpha)
        }
    
    return results


def get_modality_focus(alpha: float) -> str:
    """Get human-readable description of modality focus."""
    if alpha == 0.0:
        return "text_only"
    elif alpha < 0.5:
        return "text_heavy"
    elif alpha == 0.5:
        return "balanced"
    elif alpha < 1.0:
        return "image_heavy"
    else:
        return "image_only"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    a_norm = normalize_embedding(a)
    b_norm = normalize_embedding(b)
    return np.dot(a_norm, b_norm)


def embedding_similarity_analysis(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray
) -> dict:
    """
    Analyze similarity between text and image embeddings.
    
    Returns statistics about embedding alignment.
    """
    similarities = []
    
    for i in range(len(text_embeddings)):
        sim = cosine_similarity(text_embeddings[i], image_embeddings[i])
        similarities.append(sim)
    
    similarities = np.array(similarities)
    
    return {
        "mean_similarity": float(np.mean(similarities)),
        "std_similarity": float(np.std(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "median_similarity": float(np.median(similarities))
    }

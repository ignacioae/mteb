"""
SLERP-based multimodal adapter for meb evaluations.
Uses spherical interpolation for optimal text/image fusion.
"""

import numpy as np
import logging
from typing import List, Tuple
from .dataset_utils import slerp

logger = logging.getLogger(__name__)


class SLERPMultimodalAdapter:
    """
    SLERP-based multimodal adapter that uses spherical interpolation for text/image fusion.
    
    Architecture:
    1. For each query: query_fused = slerp(query_text, query_image, alpha)
    2. For each catalog item: catalog_fused = slerp(catalog_text, catalog_image, beta)
    3. Retrieval: cosine_similarity(query_fused, catalog_fused)
    """
    
    def __init__(self, catalog_text_embeddings: np.ndarray, catalog_image_embeddings: np.ndarray):
        """
        Initialize the SLERP multimodal adapter.
        
        Args:
            catalog_text_embeddings: Array of catalog text embeddings (n_items, text_dim)
            catalog_image_embeddings: Array of catalog image embeddings (n_items, image_dim)
        """
        self.catalog_text_embeddings = catalog_text_embeddings
        self.catalog_image_embeddings = catalog_image_embeddings
        
        # Validate dimensions
        if len(catalog_text_embeddings) != len(catalog_image_embeddings):
            raise ValueError("Text and image embeddings must have same number of items")
        
        self.n_items = len(catalog_text_embeddings)
        logger.info(f"SLERP adapter initialized with {self.n_items} catalog items")
        logger.info(f"Text embedding dim: {catalog_text_embeddings.shape[1]}")
        logger.info(f"Image embedding dim: {catalog_image_embeddings.shape[1]}")
    
    def fuse_query_embeddings(self, query_text_embedding: np.ndarray, 
                             query_image_embedding: np.ndarray, alpha: float) -> np.ndarray:
        """
        Fuse query text and image embeddings using SLERP.
        
        Args:
            query_text_embedding: Query text embedding
            query_image_embedding: Query image embedding
            alpha: Interpolation parameter (0.0 = text only, 1.0 = image only)
        
        Returns:
            Fused query embedding
        """
        return slerp(query_text_embedding, query_image_embedding, alpha)
    
    def fuse_catalog_embeddings(self, beta: float) -> np.ndarray:
        """
        Fuse all catalog text and image embeddings using SLERP.
        
        Args:
            beta: Interpolation parameter (0.0 = text only, 1.0 = image only)
        
        Returns:
            Array of fused catalog embeddings (n_items, embedding_dim)
        """
        fused_embeddings = []
        for i in range(self.n_items):
            fused = slerp(self.catalog_text_embeddings[i], self.catalog_image_embeddings[i], beta)
            fused_embeddings.append(fused)
        
        return np.array(fused_embeddings)
    
    def calculate_similarities(self, query_fused: np.ndarray, catalog_fused: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarities between query and catalog embeddings.
        
        Args:
            query_fused: Fused query embedding
            catalog_fused: Array of fused catalog embeddings
        
        Returns:
            Array of similarity scores
        """
        # Normalize embeddings for cosine similarity
        query_norm = query_fused / (np.linalg.norm(query_fused) + 1e-8)
        catalog_norm = catalog_fused / (np.linalg.norm(catalog_fused, axis=1, keepdims=True) + 1e-8)
        
        # Calculate cosine similarities
        similarities = np.dot(catalog_norm, query_norm)
        return similarities
    
    def retrieve_top_k(self, query_text_embedding: np.ndarray, query_image_embedding: np.ndarray,
                      alpha: float, beta: float, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k most similar items for a query.
        
        Args:
            query_text_embedding: Query text embedding
            query_image_embedding: Query image embedding
            alpha: Query fusion parameter
            beta: Catalog fusion parameter
            k: Number of items to retrieve
        
        Returns:
            Tuple of (indices, similarities) for top-k items
        """
        # Fuse query embeddings
        query_fused = self.fuse_query_embeddings(query_text_embedding, query_image_embedding, alpha)
        
        # Fuse catalog embeddings
        catalog_fused = self.fuse_catalog_embeddings(beta)
        
        # Calculate similarities
        similarities = self.calculate_similarities(query_fused, catalog_fused)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_similarities = similarities[top_k_indices]
        
        return top_k_indices, top_k_similarities
    
    def calculate_ndcg(self, retrieved_indices: np.ndarray, relevance_scores: List[float], 
                      k: int) -> float:
        """
        Calculate NDCG@k for retrieved items.
        
        Args:
            retrieved_indices: Indices of retrieved items
            relevance_scores: Ground truth relevance scores for all items
            k: Number of items to consider
        
        Returns:
            NDCG@k score
        """
        k = min(k, len(retrieved_indices))
        if k == 0:
            return 0.0
        
        # Get relevance scores for retrieved items
        retrieved_relevances = [relevance_scores[i] for i in retrieved_indices[:k]]
        
        # Calculate DCG
        dcg = 0.0
        for i, relevance in enumerate(retrieved_relevances):
            dcg += relevance / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        sorted_relevances = sorted(relevance_scores, reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevances):
            idcg += relevance / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg

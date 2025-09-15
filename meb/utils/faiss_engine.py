"""
FAISS-based evaluation engine for efficient similarity search.
Integrates encoders, adapters, and SLERP multimodal fusion.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class FAISSEvaluationEngine:
    """
    Evaluation engine using FAISS for efficient similarity search.
    Supports custom encoders, adapters, and caching.
    """
    
    def __init__(self, use_gpu: bool = False, cache_dir: str = "cache"):
        """
        Initialize FAISS evaluation engine.
        
        Args:
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu)
            cache_dir: Directory for caching
        """
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        self.faiss_indices = {}  # Store pre-computed indices
        self.catalog_ids = []  # Store catalog IDs for ID matching
        self.catalog_image_paths = []  # Store catalog image paths
        self.catalog_texts = []  # Store catalog texts
        
        # Try to import FAISS
        try:
            import faiss
            self.faiss = faiss
            
            if use_gpu and faiss.get_num_gpus() > 0:
                logger.info(f"FAISS GPU acceleration enabled ({faiss.get_num_gpus()} GPUs)")
            else:
                logger.info("FAISS CPU mode")
                self.use_gpu = False
                
        except ImportError:
            raise ImportError("faiss-cpu or faiss-gpu is required for FAISSEvaluationEngine")
        
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Initialized FAISS evaluation engine")
    
    def set_catalog_metadata(self, catalog_ids: List[str], catalog_image_paths: List[str], 
                           catalog_texts: List[str]):
        """
        Store catalog metadata for ID matching and retrieval analysis.
        
        Args:
            catalog_ids: List of catalog item IDs
            catalog_image_paths: List of catalog image paths
            catalog_texts: List of catalog text descriptions
        """
        self.catalog_ids = catalog_ids
        self.catalog_image_paths = catalog_image_paths
        self.catalog_texts = catalog_texts
        
        # Count instances per ID for multi-instance matching
        self.id_counts = {}
        for catalog_id in catalog_ids:
            self.id_counts[catalog_id] = self.id_counts.get(catalog_id, 0) + 1
        
        logger.info(f"Stored catalog metadata: {len(catalog_ids)} items, {len(self.id_counts)} unique IDs")
    
    def build_faiss_index(self, embeddings: np.ndarray, index_type: str = "flat") -> Any:
        """
        Build a FAISS index from embeddings.
        
        Args:
            embeddings: Embeddings array (n_items, dim)
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            
        Returns:
            FAISS index
        """
        if embeddings.size == 0:
            return None
        
        dim = embeddings.shape[1]
        n_items = embeddings.shape[0]
        
        # Ensure embeddings are float32 and C-contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        # Choose index type based on dataset size and requirements
        if index_type == "flat" or n_items < 1000:
            # Exact search for small datasets
            index = self.faiss.IndexFlatIP(dim)  # Inner product (cosine similarity for normalized vectors)
        
        elif index_type == "ivf":
            # Approximate search with IVF
            nlist = min(100, max(1, n_items // 10))  # Number of clusters
            quantizer = self.faiss.IndexFlatIP(dim)
            index = self.faiss.IndexIVFFlat(quantizer, dim, nlist)
            
            # Train the index
            index.train(embeddings)
            index.nprobe = min(10, nlist)  # Number of clusters to search
        
        elif index_type == "hnsw":
            # Hierarchical Navigable Small World
            index = self.faiss.IndexHNSWFlat(dim, 32)  # 32 is M parameter
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 100
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Move to GPU if requested
        if self.use_gpu and index_type != "hnsw":  # HNSW doesn't support GPU
            index = self.faiss.index_cpu_to_gpu(self.faiss.StandardGpuResources(), 0, index)
        
        # Add embeddings to index
        index.add(embeddings)
        
        logger.info(f"Built {index_type} FAISS index with {n_items} items, dim={dim}")
        return index
    
    def search_index(self, index: Any, query_embeddings: np.ndarray, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search FAISS index for nearest neighbors.
        
        Args:
            index: FAISS index
            query_embeddings: Query embeddings (n_queries, dim)
            k: Number of nearest neighbors to retrieve
            
        Returns:
            Tuple of (similarities, indices) arrays
        """
        if index is None or query_embeddings.size == 0:
            return np.array([]), np.array([])
        
        # Ensure query embeddings are float32 and C-contiguous
        query_embeddings = np.ascontiguousarray(query_embeddings.astype(np.float32))
        
        # Search the index
        similarities, indices = index.search(query_embeddings, k)
        
        return similarities, indices
    
    def slerp(self, v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical Linear Interpolation (SLERP) between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            t: Interpolation parameter (0.0 = v1, 1.0 = v2)
        
        Returns:
            Interpolated vector
        """
        # Handle edge cases
        if t <= 0.0:
            return v1 / (np.linalg.norm(v1) + 1e-8)
        if t >= 1.0:
            return v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Normalize input vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Calculate angle between vectors
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        
        # If vectors are very similar, use linear interpolation
        if abs(dot_product) > 0.9995:
            result = (1 - t) * v1_norm + t * v2_norm
            return result / (np.linalg.norm(result) + 1e-8)
        
        # SLERP formula
        theta = np.arccos(abs(dot_product))
        sin_theta = np.sin(theta)
        
        if sin_theta < 1e-6:
            # Vectors are nearly parallel, use linear interpolation
            result = (1 - t) * v1_norm + t * v2_norm
            return result / (np.linalg.norm(result) + 1e-8)
        
        # Apply SLERP
        factor1 = np.sin((1 - t) * theta) / sin_theta
        factor2 = np.sin(t * theta) / sin_theta
        
        if dot_product < 0:
            # Take shorter path on sphere
            factor2 = -factor2
        
        result = factor1 * v1_norm + factor2 * v2_norm
        return result / (np.linalg.norm(result) + 1e-8)
    
    def fuse_embeddings_batch(self, text_embeddings: np.ndarray, image_embeddings: np.ndarray, 
                             t: float) -> np.ndarray:
        """
        Fuse text and image embeddings using SLERP for a batch.
        
        Args:
            text_embeddings: Text embeddings (n_items, text_dim)
            image_embeddings: Image embeddings (n_items, image_dim)
            t: Interpolation parameter
            
        Returns:
            Fused embeddings (n_items, embedding_dim)
        """
        if text_embeddings.size == 0 or image_embeddings.size == 0:
            return text_embeddings if text_embeddings.size > 0 else image_embeddings
        
        fused_embeddings = []
        for i in range(len(text_embeddings)):
            fused = self.slerp(text_embeddings[i], image_embeddings[i], t)
            fused_embeddings.append(fused)
        
        return np.array(fused_embeddings)
    
    def precompute_catalog_indices(self, catalog_text_embeddings: np.ndarray, 
                                  catalog_image_embeddings: np.ndarray,
                                  beta_values: List[float],
                                  index_type: str = "flat") -> Dict[float, Any]:
        """
        Pre-compute FAISS indices for all beta values.
        
        Args:
            catalog_text_embeddings: Catalog text embeddings
            catalog_image_embeddings: Catalog image embeddings
            beta_values: List of beta values for catalog fusion
            index_type: Type of FAISS index to build
            
        Returns:
            Dictionary mapping beta values to FAISS indices
        """
        logger.info(f"Pre-computing FAISS indices for {len(beta_values)} beta values...")
        
        indices = {}
        
        for beta in beta_values:
            logger.info(f"Building index for β={beta}")
            
            # Fuse catalog embeddings
            catalog_fused = self.fuse_embeddings_batch(
                catalog_text_embeddings, catalog_image_embeddings, beta
            )
            
            # Build FAISS index
            index = self.build_faiss_index(catalog_fused, index_type)
            indices[beta] = index
        
        logger.info(f"Pre-computed {len(indices)} FAISS indices")
        self.faiss_indices = indices
        return indices
    
    def evaluate_query_batch(self, query_text_embeddings: np.ndarray, 
                            query_image_embeddings: np.ndarray,
                            alpha_values: List[float], beta_values: List[float],
                            k: int = 50) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a batch of queries across all alpha/beta combinations.
        
        Args:
            query_text_embeddings: Query text embeddings (n_queries, text_dim)
            query_image_embeddings: Query image embeddings (n_queries, image_dim)
            alpha_values: List of alpha values (query fusion)
            beta_values: List of beta values (catalog fusion)
            k: Number of nearest neighbors to retrieve
            
        Returns:
            Dictionary with results for each (alpha, beta) combination
        """
        results = {}
        n_queries = len(query_text_embeddings)
        
        # For each alpha (query fusion parameter)
        for alpha in alpha_values:
            # Fuse query embeddings once per alpha
            query_fused = self.fuse_embeddings_batch(
                query_text_embeddings, query_image_embeddings, alpha
            )
            
            # For each beta (catalog fusion parameter)
            for beta in beta_values:
                combo_key = f"alpha_{alpha}_beta_{beta}"
                
                # Get pre-computed index
                if beta not in self.faiss_indices:
                    logger.warning(f"No pre-computed index for β={beta}")
                    continue
                
                index = self.faiss_indices[beta]
                
                # Search the index
                similarities, indices = self.search_index(index, query_fused, k)
                
                # Store results
                results[combo_key] = {
                    "alpha": alpha,
                    "beta": beta,
                    "similarities": similarities,  # (n_queries, k)
                    "indices": indices,  # (n_queries, k)
                    "n_queries": n_queries,
                    "k": k
                }
        
        return results
    
    def calculate_ndcg_batch(self, retrieved_indices: np.ndarray, 
                            relevance_scores: List[List[float]], 
                            k_values: List[int]) -> Dict[str, List[float]]:
        """
        Calculate NDCG@k for a batch of queries.
        
        Args:
            retrieved_indices: Retrieved indices (n_queries, max_k)
            relevance_scores: Relevance scores for each query
            k_values: List of k values to calculate NDCG for
            
        Returns:
            Dictionary with NDCG scores for each k value
        """
        ndcg_results = {f"ndcg@{k}": [] for k in k_values}
        
        n_queries = retrieved_indices.shape[0]
        
        for query_idx in range(n_queries):
            query_relevance = relevance_scores[query_idx]
            query_indices = retrieved_indices[query_idx]
            
            for k in k_values:
                ndcg = self._calculate_single_ndcg(query_indices, query_relevance, k)
                ndcg_results[f"ndcg@{k}"].append(ndcg)
        
        return ndcg_results
    
    def _calculate_single_ndcg(self, retrieved_indices: np.ndarray, 
                              relevance_scores: List[float], k: int) -> float:
        """
        Calculate NDCG@k for a single query.
        
        Args:
            retrieved_indices: Retrieved indices for the query
            relevance_scores: Relevance scores for all catalog items (indexed by catalog position)
            k: Number of items to consider
            
        Returns:
            NDCG@k score
        """
        # Ensure k doesn't exceed catalog size or retrieved items
        k = min(k, len(retrieved_indices), len(relevance_scores))
        if k == 0:
            return 0.0
        
        # Get relevance scores for retrieved items
        retrieved_relevances = []
        for i in retrieved_indices[:k]:
            if i < len(relevance_scores):
                retrieved_relevances.append(relevance_scores[i])
            else:
                retrieved_relevances.append(0.0)  # Default relevance for out-of-bounds indices
        
        # Calculate DCG
        dcg = 0.0
        for i, relevance in enumerate(retrieved_relevances):
            dcg += relevance / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG) - only consider available items
        sorted_relevances = sorted(relevance_scores, reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevances):
            idcg += relevance / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def cleanup_indices(self):
        """Clean up FAISS indices to free memory."""
        self.faiss_indices.clear()
        logger.info("Cleaned up FAISS indices")


def create_evaluation_engine(config: Dict[str, Any]) -> FAISSEvaluationEngine:
    """
    Factory function to create evaluation engine from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized evaluation engine
    """
    return FAISSEvaluationEngine(
        use_gpu=config.get("use_gpu", False),
        cache_dir=config.get("cache_dir", "cache")
    )

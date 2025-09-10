"""
Adapter models for enhancing embeddings in MTEB evaluations.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import csv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """
    Abstract base class for embedding adapters.
    """
    
    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized {self.name} adapter")
    
    @abstractmethod
    def enhance_query_embedding(self, query_embedding: np.ndarray, **kwargs) -> np.ndarray:
        """
        Enhance a query embedding using the adapter strategy.
        
        Args:
            query_embedding: The original query embedding
            **kwargs: Additional parameters specific to the adapter
            
        Returns:
            Enhanced query embedding
        """
        pass
    
    @abstractmethod
    def calculate_enhanced_ndcg(self, relevance_scores: List[float], 
                              product_counts: List[int], k: int = 10) -> float:
        """
        Calculate enhanced NDCG considering both relevance and product availability.
        
        Args:
            relevance_scores: List of relevance scores for retrieved items
            product_counts: List of product counts for retrieved items
            k: Number of top items to consider
            
        Returns:
            Enhanced NDCG score
        """
        pass


class CatalogInterpolationAdapter(BaseAdapter):
    """
    Adapter that enhances query embeddings by interpolating with catalog centroids.
    """
    
    def __init__(self, catalog_embeddings: np.ndarray, beta: float = 0.3):
        """
        Initialize the catalog interpolation adapter.
        
        Args:
            catalog_embeddings: Array of catalog item embeddings (n_items, embedding_dim)
            beta: Interpolation parameter (0.0 = pure query, 1.0 = pure catalog)
        """
        super().__init__("CatalogInterpolationAdapter")
        self.catalog_embeddings = catalog_embeddings
        self.beta = beta
        self.catalog_centroid = np.mean(catalog_embeddings, axis=0)
        logger.info(f"Catalog centroid computed from {len(catalog_embeddings)} items")
        logger.info(f"Beta parameter set to {beta}")
    
    def enhance_query_embedding(self, query_embedding: np.ndarray, **kwargs) -> np.ndarray:
        """
        Enhance query embedding by interpolating with catalog centroid.
        
        Formula: enhanced_query = (1 - beta) * query_embedding + beta * catalog_centroid
        """
        enhanced = (1 - self.beta) * query_embedding + self.beta * self.catalog_centroid
        logger.debug(f"Enhanced query embedding with beta={self.beta}")
        return enhanced
    
    def calculate_enhanced_ndcg(self, relevance_scores: List[float], 
                              product_counts: List[int], k: int = 10) -> float:
        """
        Calculate enhanced NDCG with product count weighting.
        
        The enhanced NDCG considers both relevance and product availability:
        - Higher product counts get additional weighting
        - Uses log2(count + 1) for diminishing returns on high counts
        """
        if len(relevance_scores) != len(product_counts):
            raise ValueError("Relevance scores and product counts must have same length")
        
        k = min(k, len(relevance_scores))
        if k == 0:
            return 0.0
        
        # Calculate DCG with product count weighting
        dcg = 0.0
        for i in range(k):
            relevance = relevance_scores[i]
            count_weight = np.log2(product_counts[i] + 1)  # +1 to avoid log(0)
            enhanced_relevance = relevance * count_weight
            dcg += enhanced_relevance / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG) - sort by enhanced relevance
        enhanced_relevances = [rel * np.log2(count + 1) for rel, count in zip(relevance_scores, product_counts)]
        sorted_enhanced = sorted(enhanced_relevances, reverse=True)[:k]
        
        idcg = 0.0
        for i, enhanced_rel in enumerate(sorted_enhanced):
            idcg += enhanced_rel / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        logger.debug(f"Enhanced NDCG@{k}: {ndcg:.4f} (DCG: {dcg:.4f}, IDCG: {idcg:.4f})")
        return ndcg
    
    def set_beta(self, new_beta: float):
        """Update the beta parameter for interpolation."""
        self.beta = new_beta
        logger.info(f"Beta parameter updated to {new_beta}")


def load_catalog_data(csv_path: str) -> Tuple[List[str], List[str], List[str], List[str], 
                                           List[float], List[int], np.ndarray]:
    """
    Load catalog data from CSV file.
    
    Returns:
        Tuple of (skus, names, descriptions, categories, prices, stock_counts, embeddings)
    """
    skus, names, descriptions, categories, prices, stock_counts = [], [], [], [], [], []
    embeddings = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            skus.append(row['sku'])
            names.append(row['name'])
            descriptions.append(row['description'])
            categories.append(row['category'])
            prices.append(float(row['price']))
            stock_counts.append(int(row['stock_count']))
            
            # Parse embedding from string representation
            embedding_str = row['text_embedding'].strip('[]')
            embedding = [float(x.strip()) for x in embedding_str.split(',')]
            embeddings.append(embedding)
    
    return skus, names, descriptions, categories, prices, stock_counts, np.array(embeddings)


def load_test_queries(csv_path: str) -> Tuple[List[str], List[List[str]], List[List[int]], np.ndarray]:
    """
    Load test queries from CSV file.
    
    Returns:
        Tuple of (queries, relevant_skus_list, product_counts_list, query_embeddings)
    """
    queries, relevant_skus_list, product_counts_list = [], [], []
    query_embeddings = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row['query'])
            
            # Parse relevant SKUs
            relevant_skus = [sku.strip().strip('"') for sku in row['relevant_skus'].strip('[]').split(',')]
            relevant_skus_list.append(relevant_skus)
            
            # Parse product counts
            counts_str = row['product_counts'].strip('[]')
            product_counts = [int(x.strip()) for x in counts_str.split(',')]
            product_counts_list.append(product_counts)
            
            # Parse query embedding
            embedding_str = row['query_embedding'].strip('[]')
            embedding = [float(x.strip()) for x in embedding_str.split(',')]
            query_embeddings.append(embedding)
    
    return queries, relevant_skus_list, product_counts_list, np.array(query_embeddings)

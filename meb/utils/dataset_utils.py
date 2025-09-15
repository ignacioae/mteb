"""
Dataset utilities for loading and managing meb datasets.
Extracted from adapter_models.py to remove centroid dependencies.
"""

import os
import csv
import glob
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import requests
import io
from PIL import Image
import random

logger = logging.getLogger(__name__)


def load_catalog_data(csv_path: str) -> Tuple[List[str], List[str], List[str], List[str], np.ndarray, np.ndarray]:
    """
    Load catalog data from CSV file with standardized format.
    
    Returns:
        Tuple of (ids, texts, image_paths, metadata_list, text_embeddings, image_embeddings)
    """
    ids, texts, image_paths, metadata_list = [], [], [], []
    text_embeddings, image_embeddings = [], []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row['id'])
            texts.append(row['text'])
            image_paths.append(row['image_path'])
            metadata_list.append(row['metadata'])
            
            # Parse text embedding from string representation
            if 'text_embedding' in row and row['text_embedding']:
                text_embedding_str = row['text_embedding'].strip('[]')
                text_embedding = [float(x.strip()) for x in text_embedding_str.split(',')]
                text_embeddings.append(text_embedding)
            else:
                text_embeddings.append([])
            
            # Parse image embedding from string representation
            if 'image_embedding' in row and row['image_embedding']:
                image_embedding_str = row['image_embedding'].strip('[]')
                image_embedding = [float(x.strip()) for x in image_embedding_str.split(',')]
                image_embeddings.append(image_embedding)
            else:
                image_embeddings.append([])
    
    # Convert to numpy arrays, handling empty embeddings
    text_embeddings_array = np.array(text_embeddings) if text_embeddings and text_embeddings[0] else np.array([])
    image_embeddings_array = np.array(image_embeddings) if image_embeddings and image_embeddings[0] else np.array([])
    
    return ids, texts, image_paths, metadata_list, text_embeddings_array, image_embeddings_array


def load_test_queries(csv_path: str) -> Tuple[List[str], List[str], List[str], List[str], 
                                           List[int], np.ndarray, np.ndarray]:
    """
    Load test queries from CSV file with standardized format.
    Automatically generates product_counts if missing.
    
    Returns:
        Tuple of (ids, texts, image_paths, metadata_list, product_counts, text_embeddings, image_embeddings)
    """
    ids, texts, image_paths, metadata_list, product_counts = [], [], [], [], []
    text_embeddings, image_embeddings = [], []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check if product_counts column exists
        has_product_counts = 'product_counts' in reader.fieldnames
        if not has_product_counts:
            logger.info(f"product_counts column not found in {csv_path}, will generate random relevance scores")
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        for row in reader:
            ids.append(row['id'])
            texts.append(row['text'])
            image_paths.append(row['image_path'])
            metadata_list.append(row['metadata'])
            
            # Handle product_counts - use existing value or generate random one
            if has_product_counts:
                product_counts.append(int(row['product_counts']))
            else:
                # Generate random relevance score between 1 and 20
                product_counts.append(np.random.randint(1, 21))
            
            # Parse text embedding from string representation
            if 'text_embedding' in row and row['text_embedding']:
                text_embedding_str = row['text_embedding'].strip('[]')
                text_embedding = [float(x.strip()) for x in text_embedding_str.split(',')]
                text_embeddings.append(text_embedding)
            else:
                text_embeddings.append([])
            
            # Parse image embedding from string representation
            if 'image_embedding' in row and row['image_embedding']:
                image_embedding_str = row['image_embedding'].strip('[]')
                image_embedding = [float(x.strip()) for x in image_embedding_str.split(',')]
                image_embeddings.append(image_embedding)
            else:
                image_embeddings.append([])
    
    # Convert to numpy arrays, handling empty embeddings
    text_embeddings_array = np.array(text_embeddings) if text_embeddings and text_embeddings[0] else np.array([])
    image_embeddings_array = np.array(image_embeddings) if image_embeddings and image_embeddings[0] else np.array([])
    
    if not has_product_counts:
        logger.info(f"Generated product_counts for {len(product_counts)} test queries (range: {min(product_counts)}-{max(product_counts)})")
    
    return ids, texts, image_paths, metadata_list, product_counts, text_embeddings_array, image_embeddings_array


def load_dataset(dataset_name: str, catalog_file: str = None, test_file: str = None, 
                base_path: str = None) -> Tuple[Tuple, Tuple]:
    """
    Load a complete dataset with catalog and test data using the standardized structure.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'fashion_ecommerce', 'electronics')
        catalog_file: Specific catalog file to load (e.g., 'catalog_data.csv', 'summer_collection.csv')
                     If None, loads the first available catalog file
        test_file: Specific test file to load (e.g., 'test_queries.csv', 'seasonal_queries.csv')
                  If None, loads the first available test file
        base_path: Base path to datasets directory. If None, uses default meb/datasets
    
    Returns:
        Tuple of (catalog_data, test_data) where:
        - catalog_data: (ids, texts, image_paths, metadata_list, text_embeddings, image_embeddings)
        - test_data: (ids, texts, image_paths, metadata_list, product_counts, text_embeddings, image_embeddings)
    """
    if base_path is None:
        # Get the directory of this file and construct path to datasets
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(os.path.dirname(current_dir), 'datasets')
    
    dataset_path = os.path.join(base_path, dataset_name)
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset '{dataset_name}' not found at {dataset_path}")
    
    catalog_dir = os.path.join(dataset_path, 'catalog')
    test_dir = os.path.join(dataset_path, 'test')
    
    # Load catalog data
    if catalog_file is None:
        # Find first available catalog file
        catalog_files = glob.glob(os.path.join(catalog_dir, '*.csv'))
        if not catalog_files:
            raise ValueError(f"No catalog files found in {catalog_dir}")
        catalog_path = catalog_files[0]
        logger.info(f"Auto-selected catalog file: {os.path.basename(catalog_path)}")
    else:
        catalog_path = os.path.join(catalog_dir, catalog_file)
        if not os.path.exists(catalog_path):
            raise ValueError(f"Catalog file '{catalog_file}' not found in {catalog_dir}")
    
    # Load test data
    if test_file is None:
        # Find first available test file
        test_files = glob.glob(os.path.join(test_dir, '*.csv'))
        if not test_files:
            raise ValueError(f"No test files found in {test_dir}")
        test_path = test_files[0]
        logger.info(f"Auto-selected test file: {os.path.basename(test_path)}")
    else:
        test_path = os.path.join(test_dir, test_file)
        if not os.path.exists(test_path):
            raise ValueError(f"Test file '{test_file}' not found in {test_dir}")
    
    logger.info(f"Loading dataset '{dataset_name}' with catalog: {os.path.basename(catalog_path)}, test: {os.path.basename(test_path)}")
    
    catalog_data = load_catalog_data(catalog_path)
    test_data = load_test_queries(test_path)
    
    return catalog_data, test_data


def list_available_datasets(base_path: str = None) -> Dict[str, Dict[str, List[str]]]:
    """
    List all available datasets and their catalog/test files.
    
    Args:
        base_path: Base path to datasets directory. If None, uses default mteb/datasets
    
    Returns:
        Dictionary with dataset names as keys and dict of catalog/test files as values
    """
    if base_path is None:
        # Get the directory of this file and construct path to datasets
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(os.path.dirname(current_dir), 'datasets')
    
    datasets = {}
    
    if not os.path.exists(base_path):
        logger.warning(f"Datasets directory not found: {base_path}")
        return datasets
    
    for dataset_name in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        
        catalog_dir = os.path.join(dataset_path, 'catalog')
        test_dir = os.path.join(dataset_path, 'test')
        
        catalog_files = []
        test_files = []
        
        if os.path.exists(catalog_dir):
            catalog_files = [os.path.basename(f) for f in glob.glob(os.path.join(catalog_dir, '*.csv'))]
        
        if os.path.exists(test_dir):
            test_files = [os.path.basename(f) for f in glob.glob(os.path.join(test_dir, '*.csv'))]
        
        if catalog_files or test_files:
            datasets[dataset_name] = {
                'catalog': sorted(catalog_files),
                'test': sorted(test_files)
            }
    
    return datasets


def slerp(v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical Linear Interpolation (SLERP) between two vectors.
    
    Args:
        v1: First vector (normalized)
        v2: Second vector (normalized)
        t: Interpolation parameter (0.0 = v1, 1.0 = v2)
    
    Returns:
        Interpolated vector
    """
    # Normalize input vectors
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    
    # Handle edge cases
    if t <= 0.0:
        return v1_norm
    if t >= 1.0:
        return v2_norm
    
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


def download_image(url: str) -> Image.Image:
    user_agents=[
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36',
        'Mozilla/5.0 (Linux; Android 13; SM-S901B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36',
        'Mozilla/5.0 (Linux; Android 13; Pixel 7 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36',
        'Mozilla/5.0 (iPhone14,3; U; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/602.1.50 (KHTML, like Gecko) Version/10.0 Mobile/19A346 Safari/602.1',
        'Mozilla/5.0 (iPhone13,2; U; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/602.1.50 (KHTML, like Gecko) Version/10.0 Mobile/15E148 Safari/602.1',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246',
        'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1'
        ]
    
    response = requests.get(url, headers={'User-Agent': random.choice(user_agents)})
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content)).convert('RGB')
    
    return img
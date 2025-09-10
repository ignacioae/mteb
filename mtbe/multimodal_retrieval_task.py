"""
Multimodal retrieval task with spherical interpolation and FAISS indexing.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import faiss

from mtbe.abstasks.AbsTaskMultimodalRetrieval import AbsTaskMultimodalRetrieval
from mtbe.abstasks.TaskMetadata import TaskMetadata
from mtbe.spherical_interpolation import (
    batch_spherical_interpolation,
    interpolate_embeddings_with_alpha_sweep,
    embedding_similarity_analysis
)

logger = logging.getLogger(__name__)


class ParameterizedMultimodalRetrieval(AbsTaskMultimodalRetrieval):
    """
    Parameterized multimodal retrieval task with spherical interpolation.
    
    Supports:
    - Custom catalog and test datasets
    - Pre-computed embeddings with optional adapter
    - Alpha sweep for interpolation analysis
    - FAISS indexing for efficient retrieval
    """
    
    def __init__(
        self,
        catalog_path: str,
        test_path: str,
        task_name: Optional[str] = None,
        adapter_model=None,
        alpha_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    ):
        self.catalog_path = catalog_path
        self.test_path = test_path
        self.adapter_model = adapter_model
        self.alpha_values = alpha_values
        
        # Generate task name
        catalog_name = Path(catalog_path).stem
        test_name = Path(test_path).stem
        adapter_name = getattr(adapter_model, 'name', 'no_adapter') if adapter_model else 'no_adapter'
        
        self.task_name = task_name or f"MultimodalRetrieval_{catalog_name}_{test_name}_{adapter_name}"
        
        # Create metadata
        self.metadata = TaskMetadata(
            name=self.task_name,
            description=f"Multimodal retrieval with spherical interpolation - Catalog: {catalog_name}, Test: {test_name}",
            reference="https://github.com/cline_version/multimodal_retrieval",
            dataset={
                "catalog_path": catalog_path,
                "test_path": test_path,
                "revision": "main"
            },
            type="Retrieval",
            category="multimodal",
            modalities=["text", "image"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="ndcg@10",
            date=("2024-01-01", "2024-12-31"),
            domains=["Fashion", "E-commerce"],
            task_subtypes=["Multimodal retrieval"],
            license="mit",
            annotations_creators="derived",
            dialect=[],
            text_creation="found",
            bibtex_citation="""@misc{multimodal2024retrieval,
    title={Multimodal Retrieval with Spherical Interpolation},
    author={Cline Version},
    year={2024}
}"""
        )
        
        # Initialize data containers
        self.catalog_data = None
        self.test_data = None
        self.faiss_indices = {}
        self.evaluation_results = {}
        
        super().__init__()
    
    def load_data(self):
        """Load catalog and test data from CSV files."""
        logger.info(f"Loading catalog data from {self.catalog_path}")
        self.catalog_data = pd.read_csv(self.catalog_path)
        
        logger.info(f"Loading test data from {self.test_path}")
        self.test_data = pd.read_csv(self.test_path)
        
        # Validate required columns
        required_cols = ['text_embedding', 'image_embedding']
        for col in required_cols:
            if col not in self.catalog_data.columns:
                raise ValueError(f"Missing column '{col}' in catalog data")
            if col not in self.test_data.columns:
                raise ValueError(f"Missing column '{col}' in test data")
        
        logger.info(f"Loaded {len(self.catalog_data)} catalog items and {len(self.test_data)} test queries")
    
    def _parse_embedding(self, embedding_str: str) -> np.ndarray:
        """Parse embedding string to numpy array."""
        if isinstance(embedding_str, str):
            # Remove brackets and split by whitespace
            embedding_str = embedding_str.strip('[]')
            values = embedding_str.split()
            return np.array([float(x) for x in values])
        return np.array(embedding_str)
    
    def _load_embeddings(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Load and parse text and image embeddings from dataframe."""
        text_embeddings = []
        image_embeddings = []
        
        for _, row in data.iterrows():
            text_emb = self._parse_embedding(row['text_embedding'])
            image_emb = self._parse_embedding(row['image_embedding'])
            
            text_embeddings.append(text_emb)
            image_embeddings.append(image_emb)
        
        return np.array(text_embeddings), np.array(image_embeddings)
    
    def _apply_adapter(self, text_embeddings: np.ndarray, image_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply adapter transformations if specified."""
        if self.adapter_model is None:
            return text_embeddings, image_embeddings
        
        logger.info(f"Applying adapter: {getattr(self.adapter_model, 'name', 'unknown')}")
        
        # Apply adapter transformations
        adapted_text_embeddings = []
        adapted_image_embeddings = []
        
        for i in range(len(text_embeddings)):
            if hasattr(self.adapter_model, 'adapt_text_embedding'):
                adapted_text = self.adapter_model.adapt_text_embedding(
                    text_embeddings[i], 
                    task_name=self.task_name
                )
                adapted_text_embeddings.append(adapted_text)
            else:
                adapted_text_embeddings.append(text_embeddings[i])
            
            if hasattr(self.adapter_model, 'adapt_image_embedding'):
                adapted_image = self.adapter_model.adapt_image_embedding(
                    image_embeddings[i], 
                    task_name=self.task_name
                )
                adapted_image_embeddings.append(adapted_image)
            else:
                adapted_image_embeddings.append(image_embeddings[i])
        
        return np.array(adapted_text_embeddings), np.array(adapted_image_embeddings)
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create FAISS index for efficient similarity search."""
        dimension = embeddings.shape[1]
        
        # Convert to float32 and normalize embeddings for cosine similarity
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Create index (Inner Product for normalized vectors = cosine similarity)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        return index
    
    def _search_faiss(self, index: faiss.Index, query_embeddings: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search FAISS index with query embeddings."""
        # Normalize query embeddings
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)
        
        # Search
        similarities, indices = index.search(query_embeddings, k)
        
        return similarities, indices
    
    def _calculate_metrics(self, similarities: np.ndarray, indices: np.ndarray, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Calculate retrieval metrics."""
        metrics = {}
        
        # For this implementation, we'll use a simple relevance model
        # In practice, you would have ground truth relevance judgments
        
        for k in k_values:
            if k > indices.shape[1]:
                continue
            
            # Calculate NDCG@k (simplified - assumes top results are relevant)
            ndcg_scores = []
            precision_scores = []
            
            for i in range(len(indices)):
                # Simple relevance: higher similarity = more relevant
                relevance_scores = similarities[i][:k]
                
                # NDCG calculation (simplified)
                dcg = np.sum(relevance_scores / np.log2(np.arange(2, k + 2)))
                ideal_dcg = np.sum(np.sort(relevance_scores)[::-1] / np.log2(np.arange(2, k + 2)))
                ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
                ndcg_scores.append(ndcg)
                
                # Precision@k (simplified - assumes similarity > threshold means relevant)
                relevant_count = np.sum(relevance_scores > 0.5)
                precision = relevant_count / k
                precision_scores.append(precision)
            
            metrics[f"ndcg@{k}"] = float(np.mean(ndcg_scores))
            metrics[f"precision@{k}"] = float(np.mean(precision_scores))
        
        return metrics
    
    def evaluate_alpha_sweep(self) -> Dict[str, Any]:
        """Evaluate retrieval performance across multiple alpha values."""
        if self.catalog_data is None or self.test_data is None:
            self.load_data()
        
        # Load embeddings
        catalog_text_emb, catalog_image_emb = self._load_embeddings(self.catalog_data)
        test_text_emb, test_image_emb = self._load_embeddings(self.test_data)
        
        # Apply adapter if specified
        catalog_text_emb, catalog_image_emb = self._apply_adapter(catalog_text_emb, catalog_image_emb)
        test_text_emb, test_image_emb = self._apply_adapter(test_text_emb, test_image_emb)
        
        # Analyze embedding similarity
        similarity_analysis = embedding_similarity_analysis(catalog_text_emb, catalog_image_emb)
        
        results = {
            "task_name": self.task_name,
            "catalog_dataset": self.catalog_path,
            "test_dataset": self.test_path,
            "adapter_used": getattr(self.adapter_model, 'name', 'no_adapter') if self.adapter_model else 'no_adapter',
            "catalog_size": len(self.catalog_data),
            "test_size": len(self.test_data),
            "embedding_dimension": catalog_text_emb.shape[1],
            "embedding_similarity_analysis": similarity_analysis,
            "alpha_sweep_results": {},
            "evaluation_time": 0.0
        }
        
        start_time = time.time()
        
        for alpha in self.alpha_values:
            logger.info(f"Evaluating alpha = {alpha}")
            
            # Interpolate embeddings
            catalog_interpolated = batch_spherical_interpolation(
                catalog_text_emb, catalog_image_emb, alpha
            )
            test_interpolated = batch_spherical_interpolation(
                test_text_emb, test_image_emb, alpha
            )
            
            # Create FAISS index
            index = self._create_faiss_index(catalog_interpolated)
            
            # Search
            similarities, indices = self._search_faiss(index, test_interpolated, k=10)
            
            # Calculate metrics
            metrics = self._calculate_metrics(similarities, indices)
            
            # Store results
            results["alpha_sweep_results"][f"alpha_{alpha}"] = {
                "interpolation_alpha": alpha,
                "modality_focus": self._get_modality_focus(alpha),
                "scores": metrics,
                "mean_similarity": float(np.mean(similarities)),
                "std_similarity": float(np.std(similarities))
            }
        
        results["evaluation_time"] = time.time() - start_time
        
        # Find best alpha
        best_alpha, best_score = self._find_best_alpha(results["alpha_sweep_results"])
        results["best_alpha"] = best_alpha
        results["best_ndcg@10"] = best_score
        
        self.evaluation_results = results
        return results
    
    def _get_modality_focus(self, alpha: float) -> str:
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
    
    def _find_best_alpha(self, alpha_results: Dict) -> Tuple[float, float]:
        """Find the alpha value with the best NDCG@10 score."""
        best_alpha = 0.0
        best_score = 0.0
        
        for alpha_key, result in alpha_results.items():
            score = result["scores"].get("ndcg@10", 0.0)
            if score > best_score:
                best_score = score
                best_alpha = result["interpolation_alpha"]
        
        return best_alpha, best_score
    
    def save_results(self, output_path: str):
        """Save evaluation results to JSON file."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results to save. Run evaluate_alpha_sweep() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get summary of results as pandas DataFrame."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_alpha_sweep() first.")
        
        summary_data = []
        
        for alpha_key, result in self.evaluation_results["alpha_sweep_results"].items():
            row = {
                "alpha": result["interpolation_alpha"],
                "modality_focus": result["modality_focus"],
                **result["scores"],
                "mean_similarity": result["mean_similarity"]
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)

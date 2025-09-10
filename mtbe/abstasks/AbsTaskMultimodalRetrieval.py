from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from mtbe.abstasks.AbsTask import AbsTask, ScoresDict
from mtbe.encoder_interface import Encoder, MultimodalEncoder, AdapterEncoder
from mtbe.utils import resolve_image_path, batch_resolve_image_paths

logger = logging.getLogger(__name__)


class AbsTaskMultimodalRetrieval(AbsTask):
    """Abstract base class for multimodal retrieval tasks."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.queries = []
        self.corpus = []
        self.qrels = {}
    
    def load_data(self, **kwargs):
        """Load the dataset from the specified path."""
        if self.data_loaded:
            return
        
        data_path = Path(self.metadata.dataset["path"])
        logger.info(f"Loading data from {data_path}")
        
        try:
            data = self._load_data_from_path(data_path)
            
            self.queries = data.get('queries', [])
            self.corpus = data.get('corpus', [])
            self.qrels = data.get('qrels', {})
            
            self.data_loaded = True
            logger.info(f"Loaded {len(self.queries)} queries, {len(self.corpus)} corpus items")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def evaluate(
        self,
        model: Encoder,
        split: str = "test",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[str, Any]:
        """Evaluate a model on this task."""
        if not self.data_loaded:
            self.load_data()
        
        if split not in self.qrels:
            raise ValueError(f"Split '{split}' not found in qrels. Available splits: {list(self.qrels.keys())}")
        
        # Import here to avoid circular imports
        from mtbe.evaluation.faiss_retrieval_evaluator import FAISSRetrievalEvaluator
        
        evaluator = FAISSRetrievalEvaluator()
        scores = evaluator.evaluate(
            model=model,
            task=self,
            split=split,
            encode_kwargs=encode_kwargs,
            **kwargs
        )
        
        return [{"hf_subset": "default", **scores}]
    
    def _get_queries_for_split(self, split: str) -> list[dict]:
        """Get queries for a specific split."""
        # For simplicity, return all queries for any split
        # In a real implementation, you might filter by split
        return self.queries
    
    def _get_corpus_for_split(self, split: str) -> list[dict]:
        """Get corpus for a specific split."""
        # For simplicity, return all corpus items for any split
        return self.corpus
    
    def _get_qrels_for_split(self, split: str) -> dict:
        """Get qrels for a specific split."""
        return self.qrels.get(split, {})
    
    def _encode_queries(
        self, 
        model: Encoder, 
        queries: list[dict],
        encode_kwargs: dict[str, Any] = {}
    ) -> np.ndarray:
        """Encode queries using the model or use pre-computed embeddings."""
        # Check if queries have pre-computed embeddings
        if all("text_embedding" in q for q in queries):
            logger.info("Using pre-computed text embeddings for queries")
            embeddings = []
            for q in queries:
                embedding = np.array(q["text_embedding"])
                if isinstance(model, AdapterEncoder):
                    # Apply adapter transformation
                    embedding = model.adapt_text_embedding(
                        embedding,
                        task_name=self.metadata.name,
                        **encode_kwargs
                    )
                embeddings.append(embedding)
            return np.vstack(embeddings)
        
        # Fallback to encoding from text
        if isinstance(model, MultimodalEncoder):
            # Extract text from queries
            query_texts = [q.get("text", q.get("query", "")) for q in queries]
            return model.encode_text(
                query_texts, 
                task_name=self.metadata.name,
                **encode_kwargs
            )
        else:
            # Fallback for regular encoders
            query_texts = [q.get("text", q.get("query", "")) for q in queries]
            return model.encode(
                query_texts,
                task_name=self.metadata.name,
                **encode_kwargs
            )
    
    def _encode_corpus(
        self, 
        model: Encoder, 
        corpus: list[dict],
        encode_kwargs: dict[str, Any] = {}
    ) -> np.ndarray:
        """Encode corpus using the model."""
        if isinstance(model, MultimodalEncoder):
            # Check if corpus has images
            if any("image_path" in item for item in corpus):
                return self._encode_multimodal_corpus(model, corpus, encode_kwargs)
            else:
                # Text-only corpus
                corpus_texts = [item.get("text", item.get("title", "")) for item in corpus]
                return model.encode_text(
                    corpus_texts,
                    task_name=self.metadata.name,
                    **encode_kwargs
                )
        else:
            # Fallback for regular encoders
            corpus_texts = [item.get("text", item.get("title", "")) for item in corpus]
            return model.encode(
                corpus_texts,
                task_name=self.metadata.name,
                **encode_kwargs
            )
    
    def _encode_multimodal_corpus(
        self,
        model: MultimodalEncoder,
        corpus: list[dict],
        encode_kwargs: dict[str, Any] = {}
    ) -> np.ndarray:
        """Encode multimodal corpus (text + image) or use pre-computed embeddings."""
        # Check if all items have pre-computed embeddings
        has_text_embeddings = all("text_embedding" in item for item in corpus)
        has_image_embeddings = all("image_embedding" in item for item in corpus if "image_path" in item)
        
        if has_text_embeddings and has_image_embeddings:
            logger.info("Using pre-computed embeddings for multimodal corpus")
            embeddings = []
            for item in corpus:
                if "text_embedding" in item and "image_embedding" in item:
                    # Both text and image embeddings available
                    text_emb = np.array(item["text_embedding"])
                    image_emb = np.array(item["image_embedding"])
                    
                    if isinstance(model, AdapterEncoder):
                        # Apply adapter transformation
                        combined_emb = model.adapt_multimodal_embedding(
                            text_emb,
                            image_emb,
                            task_name=self.metadata.name,
                            **encode_kwargs
                        )
                    else:
                        # Simple concatenation
                        combined_emb = np.concatenate([text_emb, image_emb])
                    
                    embeddings.append(combined_emb)
                elif "text_embedding" in item:
                    # Only text embedding available
                    text_emb = np.array(item["text_embedding"])
                    if isinstance(model, AdapterEncoder):
                        text_emb = model.adapt_text_embedding(
                            text_emb,
                            task_name=self.metadata.name,
                            **encode_kwargs
                        )
                    embeddings.append(text_emb)
            
            return np.vstack(embeddings)
        
        # Fallback to encoding from raw data
        # Separate items with and without images
        text_only_items = []
        multimodal_items = []
        
        for item in corpus:
            if "image_path" in item and item["image_path"]:
                multimodal_items.append(item)
            else:
                text_only_items.append(item)
        
        embeddings = []
        
        # Encode text-only items
        if text_only_items:
            # Check if text-only items have pre-computed embeddings
            if all("text_embedding" in item for item in text_only_items):
                logger.info("Using pre-computed text embeddings for text-only corpus items")
                text_embeddings = []
                for item in text_only_items:
                    text_emb = np.array(item["text_embedding"])
                    if isinstance(model, AdapterEncoder):
                        text_emb = model.adapt_text_embedding(
                            text_emb,
                            task_name=self.metadata.name,
                            **encode_kwargs
                        )
                    text_embeddings.append(text_emb)
                embeddings.append(np.vstack(text_embeddings))
            else:
                # Encode from text
                texts = [item.get("text", item.get("title", "")) for item in text_only_items]
                text_embeddings = model.encode_text(
                    texts,
                    task_name=self.metadata.name,
                    **encode_kwargs
                )
                embeddings.append(text_embeddings)
        
        # Encode multimodal items
        if multimodal_items:
            texts = [item.get("text", item.get("title", "")) for item in multimodal_items]
            image_paths = [item["image_path"] for item in multimodal_items]
            
            # Resolve image paths (download URLs if needed)
            resolved_images = batch_resolve_image_paths(image_paths)
            
            # Check if model supports multimodal encoding
            if hasattr(model, 'encode_multimodal'):
                multimodal_embeddings = model.encode_multimodal(
                    texts,
                    resolved_images,
                    task_name=self.metadata.name,
                    **encode_kwargs
                )
            else:
                # Fallback: concatenate text and image embeddings
                text_embeddings = model.encode_text(
                    texts,
                    task_name=self.metadata.name,
                    **encode_kwargs
                )
                image_embeddings = model.encode_image(
                    resolved_images,
                    task_name=self.metadata.name,
                    **encode_kwargs
                )
                multimodal_embeddings = np.concatenate(
                    [text_embeddings, image_embeddings], axis=1
                )
            
            embeddings.append(multimodal_embeddings)
        
        # Combine all embeddings
        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return np.vstack(embeddings)

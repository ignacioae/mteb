from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mtbe.encoder_interface import Encoder
from mtbe.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)

# Type definitions
ScoresDict = Dict[str, Any]


class AbsTask(ABC):
    """Abstract base class for all tasks."""
    
    metadata: TaskMetadata
    
    def __init__(self, **kwargs):
        """Initialize the task."""
        self.hf_subsets = kwargs.get("hf_subsets", None)
        self.eval_splits = self.metadata.eval_splits.copy()
        self.save_suffix = ""
        
        # Data storage
        self.data_loaded = False
        self.dataset = {}
        
    @property
    def metadata_dict(self) -> dict[str, Any]:
        """Get metadata as dictionary."""
        return self.metadata.to_dict()
    
    @property
    def is_multilingual(self) -> bool:
        """Check if task is multilingual."""
        return len(self.metadata.eval_langs) > 1
    
    @property
    def languages(self) -> list[str]:
        """Get task languages."""
        return self.metadata.eval_langs
    
    @property
    def modalities(self) -> list[str]:
        """Get task modalities."""
        return self.metadata.modalities
    
    @property
    def is_aggregate(self) -> bool:
        """Check if this is an aggregate task."""
        return False
    
    @property
    def superseded_by(self) -> str | None:
        """Check if task is superseded by another."""
        return None
    
    def filter_languages(
        self, 
        languages: list[str] | None = None,
        script: list[str] | None = None,
        hf_subsets: list[str] | None = None,
        exclusive_language_filter: bool = False,
    ) -> AbsTask:
        """Filter task by languages."""
        if languages is None and script is None and hf_subsets is None:
            return self
        
        # For now, return self - in full implementation would filter subsets
        return self
    
    def filter_eval_splits(self, eval_splits: list[str] | None = None) -> AbsTask:
        """Filter evaluation splits."""
        if eval_splits is not None:
            self.eval_splits = [split for split in eval_splits if split in self.metadata.eval_splits]
        return self
    
    def filter_modalities(
        self, 
        modalities: list[str], 
        exclusive_modality_filter: bool = False
    ) -> AbsTask:
        """Filter by modalities."""
        # For now, return self - in full implementation would filter
        return self
    
    def check_if_dataset_is_superseded(self):
        """Check if dataset is superseded."""
        if self.superseded_by is not None:
            logger.warning(f"Task {self.metadata.name} is superseded by {self.superseded_by}")
    
    @abstractmethod
    def load_data(self, **kwargs):
        """Load the dataset."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        model: Encoder,
        split: str = "test",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[str, Any]:
        """Evaluate a model on this task."""
        pass
    
    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: dict[str, Any],
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        """Evaluate on a specific subset."""
        raise NotImplementedError("Subclasses must implement _evaluate_subset")
    
    def _load_data_from_path(self, data_path: str | Path) -> dict[str, Any]:
        """Load data from a local path."""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist")
        
        # Load queries
        queries_file = data_path / "queries.jsonl"
        corpus_file = data_path / "corpus.jsonl"
        qrels_dir = data_path / "qrels"
        
        data = {}
        
        # Load queries
        if queries_file.exists():
            with open(queries_file, 'r', encoding='utf-8') as f:
                data['queries'] = [json.loads(line) for line in f]
        
        # Load corpus
        if corpus_file.exists():
            with open(corpus_file, 'r', encoding='utf-8') as f:
                data['corpus'] = [json.loads(line) for line in f]
        
        # Load qrels for each split
        if qrels_dir.exists():
            data['qrels'] = {}
            for qrels_file in qrels_dir.glob("*.tsv"):
                split_name = qrels_file.stem
                qrels = {}
                with open(qrels_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            query_id, doc_id, relevance = parts[0], parts[1], int(parts[2])
                            if query_id not in qrels:
                                qrels[query_id] = {}
                            qrels[query_id][doc_id] = relevance
                data['qrels'][split_name] = qrels
        
        return data

from __future__ import annotations

from mtbe.abstasks.AbsTaskMultimodalRetrieval import AbsTaskMultimodalRetrieval
from mtbe.abstasks.TaskMetadata import TaskMetadata


class SampleImageTextRetrieval(AbsTaskMultimodalRetrieval):
    """Sample multimodal retrieval task for testing the framework."""
    
    metadata = TaskMetadata(
        name="SampleImageTextRetrieval",
        description="A sample task for text-to-image retrieval evaluation using synthetic data",
        reference="https://github.com/your-repo/cline_version",
        dataset={
            "path": "cline_version/datasets/sample_dataset",
            "revision": "main"
        },
        type="Retrieval",
        category="t2t",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs=["eng"],
        main_score="ndcg@100",
        date=("2024-01-01", "1.0.0"),
        domains=["Academic"],
        license="MIT",
        text_creation="synthetic",
        n_samples={"test": 50},
        avg_character_length={"queries": 50.0, "corpus": 100.0},
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

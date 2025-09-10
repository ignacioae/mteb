"""
Sample multimodal retrieval task with pre-computed embeddings.
"""

from mtbe.abstasks.AbsTaskMultimodalRetrieval import AbsTaskMultimodalRetrieval
from mtbe.abstasks.TaskMetadata import TaskMetadata


class SamplePrecomputedImageTextRetrieval(AbsTaskMultimodalRetrieval):
    """Sample multimodal retrieval task that uses pre-computed embeddings."""
    
    metadata = TaskMetadata(
        name="SamplePrecomputedImageTextRetrieval",
        description="Sample multimodal retrieval task with pre-computed text and image embeddings",
        reference="https://github.com/example/sample-precomputed-dataset",
        dataset={
            "path": "cline_version/datasets/sample_dataset_with_embeddings/",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg@100",
        date=("2024-01-01", "2024-12-31"),
        domains=["Academic", "Non-fiction"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@misc{sample2024precomputed,
    title={Sample Precomputed Multimodal Retrieval Dataset},
    author={Sample Author},
    year={2024}
}""",
        n_samples={"test": 3},
        avg_character_length={
            "test": 45.2,
        },
    )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# Type definitions
TASK_TYPE = Literal["Retrieval", "Classification", "Clustering", "STS", "Reranking"]
TASK_CATEGORY = Literal["t2t", "s2s", "s2p", "p2p"]  # text2text, sentence2sentence, etc.
TASK_DOMAIN = Literal[
    "Academic", "Blog", "Creative", "Legal", "Medical", "News", "Reviews", 
    "Social", "Web", "Wiki", "Fiction", "Non-fiction", "Government"
]
MODALITIES = Literal["text", "image", "audio", "video"]


@dataclass
class TaskMetadata:
    """Metadata for a task."""
    
    name: str
    description: str = ""
    reference: str = ""
    dataset: dict[str, str] = field(default_factory=dict)
    type: TASK_TYPE = "Retrieval"
    category: TASK_CATEGORY = "t2t"
    modalities: list[MODALITIES] = field(default_factory=lambda: ["text"])
    eval_splits: list[str] = field(default_factory=lambda: ["test"])
    eval_langs: list[str] = field(default_factory=lambda: ["eng"])
    main_score: str = "ndcg@100"
    date: tuple[str, str] | None = None  # (creation_date, version)
    form: list[str] = field(default_factory=list)
    domains: list[TASK_DOMAIN] | None = None
    task_subtypes: list[str] = field(default_factory=list)
    license: str = ""
    socioeconomic_status: str = ""
    annotations_creators: str = ""
    dialect: list[str] = field(default_factory=list)
    text_creation: str = ""
    bibtex_citation: str = ""
    n_samples: dict[str, int] = field(default_factory=dict)
    avg_character_length: dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.description:
            self.description = f"Multimodal retrieval task: {self.name}"
        
        if not self.dataset:
            self.dataset = {
                "path": f"cline_version/datasets/{self.name.lower()}",
                "revision": "main"
            }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @property
    def languages(self) -> list[str]:
        """Alias for eval_langs for compatibility."""
        return self.eval_langs
    
    @property
    def scripts(self) -> list[str]:
        """Get script codes for languages."""
        # Simplified mapping - in real implementation would use proper ISO codes
        script_mapping = {
            "eng": "Latn",
            "spa": "Latn", 
            "fra": "Latn",
            "deu": "Latn",
            "ita": "Latn",
            "por": "Latn",
            "rus": "Cyrl",
            "ara": "Arab",
            "zho": "Hans",
            "jpn": "Jpan",
            "kor": "Kore",
        }
        return [script_mapping.get(lang, "Latn") for lang in self.eval_langs]
    
    @property
    def hf_subsets_to_langscripts(self) -> dict[str, list[str]]:
        """Map HuggingFace subsets to language-script codes."""
        return {
            "default": [f"{lang}-{script}" for lang, script in zip(self.eval_langs, self.scripts)]
        }
    
    @property
    def is_public(self) -> bool:
        """Whether the dataset is publicly available."""
        return True  # Default to public for sample datasets

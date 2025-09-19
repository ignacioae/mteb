from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import tqdm

from scipy.spatial import geometric_slerp

from mteb.encoder_interface import Encoder, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package
from mteb.interpolate_embeddings import slerp

MULTILINGUAL_EVALUATED_LANGUAGES = [
    "arb-Arab",
    "ben-Beng",
    "eng-Latn",
    "spa-Latn",
    "deu-Latn",
    "pes-Arab",
    "fin-Latn",
    "fra-Latn",
    "hin-Deva",
    "ind-Latn",
    "jpn-Jpan",
    "kor-Hang",
    "rus-Cyrl",
    "swh-Latn",
    "tel-Telu",
    "tha-Thai",
    "yor-Latn",
    "zho-Hant",
    "zho-Hans",
]

MODEL_PROMPTS = {
    "Classification": "CLASSIFICATION",
    "MultilabelClassification": "CLASSIFICATION",
    "Clustering": "CLUSTERING",
    "STS": "SEMANTIC_SIMILARITY",
    PromptType.query.value: "RETRIEVAL_QUERY",
    PromptType.document.value: "RETRIEVAL_DOCUMENT",
}

GECKO_TRAINING_DATA = {
    # Ones that are available from HF.
    "NQHardNegatives": ["train"],
    "FEVERHardNegatives": ["train"],
    "HotpotQAHardNegatives": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
}

MULTIMODAL_TRAINING_DATA = {
    "ProviderImagesDescriptions": ["train"],
    "InstagramImagesDescriptions": ["train"],
    "WebImagesAdd2CartQueries": ["train"],
}


class GoogleTextEmbeddingModel(Encoder, Wrapper):
    def __init__(
        self,
        model_name: str,
        sep: str = " ",
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

    def _embed(
        self,
        texts: list[str],
        google_task_type: str | None = None,
        show_progress_bar: bool = False,
        titles: list[str] | None = None,
        dimensionality: int | None = 768,
    ) -> list[list[float]]:
        """Embeds texts with a pre-trained, foundational model.
        From https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#generative-ai-get-text-embedding-python_vertex_ai_sdk
        """
        requires_package(
            self, "vertexai", self.model_name, "pip install 'mteb[vertexai]'"
        )
        from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

        model = TextEmbeddingModel.from_pretrained(self.model_name)
        if titles:
            # Allow title-only embeddings by replacing text with a space
            # Else Google throws google.api_core.exceptions.InvalidArgument: 400 The text content is empty.
            inputs = [
                TextEmbeddingInput(
                    text if text else " ", task_type=google_task_type, title=title
                )
                for text, title in zip(texts, titles)
            ]
        else:
            inputs = [
                TextEmbeddingInput(text, task_type=google_task_type) for text in texts
            ]

        kwargs = {"output_dimensionality": dimensionality} if dimensionality else {}

        max_batch_size = 16  ## Vertex API limits the number of instances per call to 250, but there is also a limit of tokens involved. Let's be conservative and set it to 16 by default. TODO: in a future PR, leverage the CountTokens API to get the optimum batch size for each request.
        batches = [
            inputs[i : i + max_batch_size]
            for i in range(0, len(inputs), max_batch_size)
        ]

        all_embeddings = []

        for batch in tqdm.tqdm(batches, leave=False, disable=not show_progress_bar):
            try:
                embeddings_batch = model.get_embeddings(batch, **kwargs)
            # Except the very rare google.api_core.exceptions.InternalServerError
            except Exception as e:
                print("Retrying once after error:", e)
                embeddings_batch = model.get_embeddings(batch, **kwargs)

            all_embeddings.extend([embedding.values for embedding in embeddings_batch])

        return np.asarray(all_embeddings)

    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        prompt_name = self.get_prompt_name(self.model_prompts, task_name, prompt_type)
        google_task_type = self.model_prompts.get(prompt_name)

        show_progress_bar = (
            False
            if "show_progress_bar" not in kwargs
            else kwargs.pop("show_progress_bar")
        )

        return self._embed(
            sentences,
            google_task_type=google_task_type,
            show_progress_bar=show_progress_bar,
        )


google_text_emb_004 = ModelMeta(
    loader=partial(
        GoogleTextEmbeddingModel,
        model_name="text-embedding-004",
        model_prompts=MODEL_PROMPTS,
    ),
    name="google/text-embedding-004",
    languages=["eng-Latn"],
    open_weights=False,
    revision="1",  # revision is intended for implementation
    release_date="2024-05-14",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=2048,
    embed_dim=768,
    license=None,
    reference="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
)

google_text_emb_005 = ModelMeta(
    loader=partial(
        GoogleTextEmbeddingModel,
        model_name="text-embedding-005",
        model_prompts=MODEL_PROMPTS,
    ),
    name="google/text-embedding-005",
    languages=["eng-Latn"],
    open_weights=False,
    revision="1",  # revision is intended for implementation
    release_date="2024-11-18",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=2048,
    embed_dim=768,
    license=None,
    reference="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
)

google_text_multilingual_emb_002 = ModelMeta(
    loader=partial(
        GoogleTextEmbeddingModel,
        model_name="text-multilingual-embedding-002",
        model_prompts=MODEL_PROMPTS,
    ),
    name="google/text-multilingual-embedding-002",
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,  # From the list of evaluated languages in https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#supported_text_languages
    open_weights=False,
    revision="1",
    release_date="2024-05-14",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=2048,
    embed_dim=768,
    license=None,
    reference="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
)

google_gemini_embedding_001 = ModelMeta(
    loader=partial(
        GoogleTextEmbeddingModel,
        model_name="gemini-embedding-001",
        model_prompts=MODEL_PROMPTS,
    ),
    name="google/gemini-embedding-001",
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=False,
    revision="1",
    release_date="2025-03-07",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=2048,
    embed_dim=3072,
    license=None,
    reference="https://ai.google.dev/gemini-api/docs/embeddings",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
)

embedding_gemma_300m = ModelMeta(
    name="google/embeddinggemma-300m",
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2",
    release_date="2025-09-04",
    n_parameters=307_581_696,
    embed_dim=768,
    max_tokens=2048,
    license="gemma",
    reference="https://ai.google.dev/gemma/docs/embeddinggemma/model_card",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    similarity_fn_name="cosine",
    memory_usage_mb=578,
)

class GoogleMultimodalEmbeddingModel(Encoder, Wrapper):
    """Google Multimodal Embedding Model for embedding text and images."""    
    def __init__(
        self,
        model_name: str,
        **kwargs,
    ) -> None:
        self.model_name = model_name

    def _embed(
        self,
        images: list[str] | None = None,
        texts: list[str] | None = None,
        alpha: float = 0.5,
        show_progress_bar: bool = False,
        dimensionality: int | None = 1408,
    ) -> np.ndarray:
        """Embeds multimodal content with a pre-trained, foundational model.
        
        Args:
            images: List of image file paths to embed
            texts: List of text strings to embed
            alpha: Interpolation weight for combining text and image embeddings (0.0 = all image, 1.0 = all text)
            show_progress_bar: Whether to show progress bar during processing
            dimensionality: Output dimensionality for embeddings
            
        Returns:
            Array of embeddings
            
        Raises:
            ValueError: If neither images nor texts are provided, or if lengths don't match when both are provided
            
        Reference:
            https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings
        """
        if not images and not texts:
            raise ValueError("At least one of 'images' or 'texts' must be provided")
            
        if images and texts and len(images) != len(texts):
            raise ValueError(f"When both images and texts are provided, they must have the same length. "
                            f"Got {len(images)} images and {len(texts)} texts")
        
        requires_package(
            self, "vertexai", self.model_name, "pip install 'mteb[vertexai]'"
        )
        from vertexai.vision_models import MultiModalEmbeddingModel, Image

        model = MultiModalEmbeddingModel.from_pretrained(self.model_name)
        max_batch_size = 32
        embed_kwargs = {"output_dimensionality": dimensionality} if dimensionality else {}
        
        all_img_embeddings = None
        all_text_embeddings = None
        
        if images:
            try:
                img_inputs = [Image.load_from_file(image) for image in images]
            except Exception as e:
                raise ValueError(f"Failed to load image files: {e}")
            
            img_batches = [
                img_inputs[i : i + max_batch_size] 
                for i in range(0, len(img_inputs), max_batch_size)
            ]
            
            all_img_embeddings = []
            for batch in tqdm.tqdm(img_batches, desc="Processing images", leave=False, disable=not show_progress_bar):
                try:
                    img_embeddings_batch = model.get_embeddings(images=batch, **embed_kwargs)
                except Exception as e:
                    print(f"Retrying image batch after error: {e}")
                    try:
                        img_embeddings_batch = model.get_embeddings(images=batch, **embed_kwargs)
                    except Exception as retry_e:
                        raise RuntimeError(f"Failed to get image embeddings after retry: {retry_e}")

                all_img_embeddings.extend([embedding.image_embedding for embedding in img_embeddings_batch])

        if texts:
            text_batches = [
                texts[i : i + max_batch_size] 
                for i in range(0, len(texts), max_batch_size)
            ]
            
            all_text_embeddings = []
            for batch in tqdm.tqdm(text_batches, desc="Processing texts", leave=False, disable=not show_progress_bar):
                try:
                    text_embeddings_batch = model.get_embeddings(contextual_text=batch, **embed_kwargs)
                except Exception as e:
                    print(f"Retrying text batch after error: {e}")
                    try:
                        text_embeddings_batch = model.get_embeddings(contextual_text=batch, **embed_kwargs)
                    except Exception as retry_e:
                        raise RuntimeError(f"Failed to get text embeddings after retry: {retry_e}")

                all_text_embeddings.extend([embedding.text_embedding for embedding in text_embeddings_batch])
                
        if texts and images:
            all_embeddings = [
                slerp(img_emb, text_emb, alpha)
                for img_emb, text_emb in zip(all_img_embeddings, all_text_embeddings)
            ]
        elif texts and not images:
            all_embeddings = all_text_embeddings
        elif images and not texts:
            all_embeddings = all_img_embeddings
            
        return np.asarray(all_embeddings)

    def encode(
        self,
        sentences: list[str] | None = None,
        images: list[str] | None = None,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode text and/or images into embeddings.
        
        Args:
            sentences: List of text strings to encode
            images: List of image file paths to encode
            alpha: Interpolation weight when both text and images are provided (0.0 = all image, 1.0 = all text)
            **kwargs: Additional keyword arguments
            
        Returns:
            Array of embeddings
        """
        show_progress_bar = kwargs.pop("show_progress_bar", False)
        dimensionality = kwargs.pop("dimensionality", 1408)

        return self._embed(
            texts=sentences,
            images=images,
            alpha=alpha,
            show_progress_bar=show_progress_bar,
            dimensionality=dimensionality,
        )


google_multimodal_embedding_001 = ModelMeta(
    loader=partial(
        GoogleMultimodalEmbeddingModel,
        model_name="multimodalembedding@001",
    ),
    name="google/multimodalembedding@001",
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=False,
    revision="1",
    release_date="2024-05-14",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=2048,
    embed_dim=1408,
    license=None,
    reference="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=MULTIMODAL_TRAINING_DATA,
)

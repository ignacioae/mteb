# meb Advanced Multimodal Evaluation System

A sophisticated evaluation system for multimodal text/image retrieval using SLERP (Spherical Linear Interpolation) fusion and advanced indexing with FAISS.

## Overview

The meb (Massive Text and Business Embedding) Advanced System provides a comprehensive framework for evaluating multimodal retrieval models with:

- **SLERP Fusion**: Spherical linear interpolation for optimal text/image combination
- **Extensible Architecture**: Support for custom encoders and adapters
- **FAISS Integration**: Efficient similarity search with GPU acceleration
- **Intelligent Caching**: Automatic caching of embeddings and transformations
- **Configuration-Driven**: JSON-based configuration for reproducible experiments

## Key Features

- **SLERP Multimodal Fusion**: Advanced spherical interpolation for text/image embeddings
- **Custom Encoders**: Support for HuggingFace models, custom models, and precomputed embeddings
- **Flexible Adapters**: PCA, normalization, rotation, scaling, and composite transformations
- **FAISS Indexing**: Multiple index types (Flat, IVF, HNSW) with GPU support
- **Comprehensive Evaluation**: NDCG@k metrics with parameter optimization
- **Standardized Dataset Format**: Support for both text and image data

## Quick Start

### Basic Evaluation (Precomputed Embeddings)

```bash
cd meb
python run_advanced_evaluation.py --config config_examples/basic_config.json
```

### SLERP Evaluation

```bash
python run_evaluation.py
```

### Custom Configuration

```bash
python run_advanced_evaluation.py --encoder mock --adapter normalize --dataset electronics
```

## Core Concepts

### SLERP Multimodal Fusion

The system uses Spherical Linear Interpolation (SLERP) for optimal multimodal fusion:

**Query Fusion:**
```
query_fused = slerp(query_text_embedding, query_image_embedding, α)
```

**Catalog Fusion:**
```
catalog_fused = slerp(catalog_text_embedding, catalog_image_embedding, β)
```

**Parameters:**
- `α = 0.0`: Text-only queries
- `α = 1.0`: Image-only queries  
- `α = 0.5`: Balanced text/image fusion
- `β = 0.0`: Text-only catalog
- `β = 1.0`: Image-only catalog
- `β = 0.5`: Balanced text/image catalog

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Encoders     │    │    Adapters     │    │ FAISS Engine   │
│                 │    │                 │    │                 │
│ • Custom Models │───▶│ • PCA           │───▶│ • GPU Support   │
│ • HuggingFace   │    │ • Normalization │    │ • Index Types   │
│ • Precomputed   │    │ • Rotation      │    │ • SLERP Fusion  │
│ • Mock/Testing  │    │ • Composite     │    │ • Batch Eval    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Dataset Format

The meb system requires datasets in CSV format with specific column structures. All datasets must follow this standardized format for proper evaluation.

### Catalog Dataset CSV Format

**Required Columns (must be present):**
- `id` (string): Unique item identifier (e.g., "TECH001", "ITEM001")
- `text` (string): Text description/content of the item
- `image_path` (string): Relative path to associated image file
- `metadata` (string): JSON-formatted metadata containing additional item information

**Optional Columns (for precomputed embeddings):**
- `text_embedding` (string): JSON array of floats representing text embedding (e.g., "[0.2, 0.8, 0.1, ...]")
- `image_embedding` (string): JSON array of floats representing image embedding (e.g., "[0.1, 0.3, 0.7, ...]")

**Column Data Types:**
- `id`: String (no spaces, alphanumeric + underscore recommended)
- `text`: String (any length, UTF-8 encoded)
- `image_path`: String (relative path from dataset root)
- `metadata`: String (valid JSON format, escaped quotes)
- `text_embedding`: String (JSON array format: "[float1, float2, ...]")
- `image_embedding`: String (JSON array format: "[float1, float2, ...]")

**Example Catalog CSV:**
```csv
id,text,image_path,metadata,text_embedding,image_embedding
TECH001,Wireless Bluetooth Headphones - High-quality wireless headphones with noise cancellation,images/headphones_001.jpg,"{""category"": ""Audio"", ""price"": 199.99, ""brand"": ""SoundTech"", ""wireless"": true}","[0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4, 0.5, 0.8]","[0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.4, 0.6, 0.7, 0.3]"
ITEM001,Summer Floral Dress - Beautiful floral dress perfect for summer occasions,images/dress_001.jpg,"{""category"": ""Dresses"", ""price"": 89.99, ""brand"": ""SummerStyle""}","[0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.5, 0.2]","[0.1, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.9, 0.1, 0.5]"
```

### Test Dataset CSV Format

**Required Columns (must be present):**
- `id` (string): Unique query identifier (e.g., "QUERY001", "TEST001")
- `text` (string): Query text describing what the user is searching for
- `image_path` (string): Relative path to query image file
- `metadata` (string): JSON-formatted metadata containing query context
- `product_counts` (integer): Relevance score or count (0 to N, higher = more relevant)

**Optional Columns (for precomputed embeddings):**
- `text_embedding` (string): JSON array of floats representing query text embedding
- `image_embedding` (string): JSON array of floats representing query image embedding

**Column Data Types:**
- `id`: String (no spaces, alphanumeric + underscore recommended)
- `text`: String (search query, any length, UTF-8 encoded)
- `image_path`: String (relative path from dataset root)
- `metadata`: String (valid JSON format, escaped quotes)
- `product_counts`: Integer (0 to N, relevance score)
- `text_embedding`: String (JSON array format: "[float1, float2, ...]")
- `image_embedding`: String (JSON array format: "[float1, float2, ...]")

**Example Test CSV:**
```csv
id,text,image_path,metadata,product_counts,text_embedding,image_embedding
QUERY001,summer dress,images/query_dress.jpg,"{""intent"": ""purchase"", ""season"": ""summer""}",15,"[0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.4, 0.6, 0.7, 0.3]","[0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.5, 0.2]"
QUERY002,wireless headphones,images/query_headphones.jpg,"{""intent"": ""research"", ""budget"": ""high""}",8,"[0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4, 0.5, 0.8]","[0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.4, 0.6, 0.7, 0.3]"
```

### Important Notes

1. **No Extra Columns**: Avoid columns like "Unnamed: 0" which can cause parsing errors
2. **Consistent Embedding Dimensions**: All embeddings in the same dataset must have the same dimensionality
3. **JSON Escaping**: Metadata must use escaped quotes: `"{""key"": ""value""}"`
4. **File Encoding**: Use UTF-8 encoding for all CSV files
5. **Path Separators**: Use forward slashes (/) in image paths for cross-platform compatibility
6. **Header Row**: First row must contain column names exactly as specified above

## Project Structure

```
meb/
├── run_evaluation.py              # SLERP evaluation script
├── run_advanced_evaluation.py     # Advanced evaluation pipeline
├── models/
│   ├── encoders.py                # Base encoder system
│   ├── fashionclip_encoder.py     # FashionCLIP encoder
│   ├── adapters.py                # Base adapter system  
│   ├── finetuning_adapter.py      # Fine-tuning adapter
│   ├── slerp_adapter.py           # SLERP multimodal adapter
│   ├── faiss_engine.py            # FAISS evaluation engine
│   └── dataset_utils.py           # Dataset loading utilities
├── config_examples/               # Configuration examples
├── datasets/                      # Sample datasets
│   ├── fashion_ecommerce/
│   └── electronics/
└── README.md                      # This file
```

## Usage Examples

### Basic SLERP Usage

```python
from models.dataset_utils import load_dataset
from models.slerp_adapter import SLERPMultimodalAdapter

# Load dataset
catalog_data, test_data = load_dataset('fashion_ecommerce')
catalog_ids, catalog_texts, _, _, catalog_text_embeddings, catalog_image_embeddings = catalog_data
test_ids, test_texts, _, _, test_counts, test_text_embeddings, test_image_embeddings = test_data

# Create SLERP adapter
adapter = SLERPMultimodalAdapter(catalog_text_embeddings, catalog_image_embeddings)

# Retrieve top-k items with SLERP fusion
alpha = 0.5  # Query fusion parameter
beta = 0.5   # Catalog fusion parameter
top_indices, similarities = adapter.retrieve_top_k(
    test_text_embeddings[0], test_image_embeddings[0], alpha, beta, k=10
)

# Calculate NDCG
relevance_scores = [1.0, 0.8, 0.6, 0.4, 0.2] * 20  # Mock relevance scores
ndcg_score = adapter.calculate_ndcg(top_indices, relevance_scores, k=5)

print(f"NDCG@5: {ndcg_score:.4f}")
print(f"Top 5 retrieved items: {top_indices[:5]}")
```

### Advanced Pipeline with Custom Encoder

```python
from models.encoders import get_encoder
from models.adapters import get_adapter
from models.faiss_engine import FAISSEvaluationEngine

# Initialize components
encoder = get_encoder({
    "type": "huggingface",
    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
    "image_model": "sentence-transformers/clip-ViT-B-32"
})

adapter = get_adapter({
    "type": "pca",
    "n_components": 128
})

engine = FAISSEvaluationEngine(use_gpu=True)

# Generate and transform embeddings
text_embeddings, image_embeddings = encoder.encode_dataset(
    texts, image_paths, "my_dataset", "catalog"
)

transformed_text, transformed_image = adapter.transform_dataset(
    text_embeddings, image_embeddings, encoder.name, "my_dataset", "catalog"
)
```

### FashionCLIP with Fine-tuning

```python
from models.fashionclip_encoder import FashionCLIPEncoder
from models.finetuning_adapter import FineTuningAdapter

# Initialize FashionCLIP encoder
encoder = FashionCLIPEncoder(
    model_path="path/to/fashionclip/model",
    device="cuda"
)

# Initialize fine-tuning adapter
adapter = FineTuningAdapter(
    method="mlp",
    input_dim=512,
    hidden_dims=[256, 128],
    output_dim=128,
    learning_rate=0.001
)

# Encode and transform
text_embeddings = encoder.encode_text(texts)
image_embeddings = encoder.encode_image(image_paths)

# Apply fine-tuning transformation
transformed_embeddings = adapter.transform_embeddings(text_embeddings)
```

## Configuration

### Basic Configuration

```json
{
  "encoder": "precomputed",
  "adapter": "identity",
  "datasets": "all",
  "alpha_values": [0.0, 0.25, 0.5, 0.75, 1.0],
  "beta_values": [0.0, 0.25, 0.5, 0.75, 1.0],
  "k_values": [1, 5, 10, 50],
  "index_type": "flat",
  "engine": {
    "use_gpu": false,
    "cache_dir": "cache"
  }
}
```

### Advanced Configuration

```json
{
  "encoder": {
    "type": "huggingface",
    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
    "image_model": "sentence-transformers/clip-ViT-B-32",
    "device": "cuda"
  },
  "adapter": [
    {"type": "normalize"},
    {"type": "pca", "n_components": 64}
  ],
  "datasets": ["fashion_ecommerce"],
  "alpha_values": [0.0, 0.5, 1.0],
  "beta_values": [0.0, 0.5, 1.0],
  "k_values": [1, 5, 10],
  "index_type": "ivf",
  "engine": {
    "use_gpu": true,
    "cache_dir": "cache/experiments"
  }
}
```

## Results

Results are automatically saved with timestamps:

- **JSON file**: Detailed results with all metrics
- **Text file**: Human-readable summary

### Sample Output

```
🎯 ADVANCED EVALUATION COMPLETE!
==================================================
📊 Key Findings:
   • Encoder: mock_128
   • Adapter: normalize
   • Datasets evaluated: 1
   • Average NDCG@5: 0.8595 ± 0.0000
   • Best parameters: α=1.0, β=0.0
```

## Performance

### Efficiency Features

1. **Pre-computed indices**: Catalog embeddings fused once per β value
2. **Batch processing**: All queries evaluated simultaneously
3. **Memory optimization**: Automatic cleanup of FAISS indices
4. **Intelligent caching**: Avoid recomputation of embeddings

### Scalability

- **Small datasets** (< 1K items): Exact search with flat index
- **Medium datasets** (1K-100K items): Approximate search with IVF
- **Large datasets** (> 100K items): HNSW index with GPU acceleration

## Adding New Datasets

1. Create directory structure: `datasets/your_dataset/catalog/` and `datasets/your_dataset/test/`
2. Add CSV files following the required format
3. Run evaluation: `python run_advanced_evaluation.py`

The system will automatically discover and evaluate new datasets.

## Requirements

### Required
- Python 3.7+
- NumPy
- `faiss-cpu` or `faiss-gpu`

### Optional
- `sentence-transformers`: HuggingFace encoder support
- `scikit-learn`: PCA adapter support
- `PIL`: Image loading for HuggingFace image models

### Installation

```bash
# CPU version
pip install faiss-cpu numpy

# GPU version (if CUDA available)
pip install faiss-gpu numpy

# Optional dependencies
pip install sentence-transformers scikit-learn pillow
```

## License

This project is part of the MTEB framework ecosystem.

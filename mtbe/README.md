# Cline Version: Multimodal Retrieval Framework

A framework for evaluating multimodal retrieval models, designed to be analogous to MTEB but optimized for text-image and multimodal retrieval tasks using FAISS for efficient similarity search.

## 🚀 Quick Start

```python
import cline_version as cv

# Load tasks and model
tasks = cv.get_tasks(tasks=["SampleImageTextRetrieval"])
model = cv.get_model("sample-multimodal-encoder")

# Run evaluation
evaluator = cv.MultimodalMTEB(tasks=tasks)
results = evaluator.run(model, output_folder="results")

# Analyze results
from cline_version.evaluation.MultimodalMTEB import BenchmarkResults
benchmark_results = BenchmarkResults("results")
benchmark_results.print_leaderboard()
```

## 🎯 Key Features

- **🔍 FAISS-based Evaluation**: Efficient similarity search for large-scale retrieval
- **🖼️ Multimodal Support**: Text, image, and combined multimodal embeddings
- **📊 MTEB-like API**: Familiar interface for MTEB users
- **🔧 Extensible Architecture**: Easy to add new tasks and models
- **📈 Comprehensive Metrics**: nDCG@100, Recall@k, MAP, and more
- **⚡ Performance Optimized**: Uses your existing FAISS-based evaluation approach

## 🏗️ Architecture

The framework follows MTEB's design patterns but is specialized for multimodal retrieval:

```
cline_version/
├── __init__.py                 # Main API (analogous to mteb.__init__.py)
├── overview.py                 # Task management (analogous to mteb.overview)
├── encoder_interface.py        # Model interfaces
├── model_meta.py              # Model metadata
├── abstasks/                  # Abstract task classes
│   ├── AbsTask.py             # Base task class
│   ├── AbsTaskMultimodalRetrieval.py  # Multimodal retrieval base
│   └── TaskMetadata.py        # Task metadata
├── tasks/                     # Concrete task implementations
│   └── sample_image_text_retrieval.py
├── models/                    # Model implementations
│   └── sample_models.py       # Sample encoders
├── evaluation/                # Evaluation pipeline
│   ├── MultimodalMTEB.py      # Main evaluator (analogous to MTEB.py)
│   └── faiss_retrieval_evaluator.py  # FAISS-based evaluation
├── benchmarks/                # Benchmark definitions
├── datasets/                  # Sample datasets with test splits
└── examples/                  # Usage examples
```

## 📦 Installation

```bash
# Clone or copy the cline_version folder
cd cline_version

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install numpy faiss-cpu
```

For GPU acceleration:
```bash
pip install faiss-gpu
```

## 🎮 Usage Examples

### Basic Evaluation

```python
import cline_version as cv

# List available tasks
cv.print_available_tasks()

# Get specific tasks
tasks = cv.get_tasks(
    task_types=["Retrieval"],
    modalities=["text", "image"]
)

# Load a model
model = cv.get_model("sample-multimodal-encoder")

# Run evaluation
evaluator = cv.MultimodalMTEB(tasks=tasks)
results = evaluator.run(
    model=model,
    output_folder="results",
    verbosity=2
)
```

### Custom Model Integration

```python
from cline_version.encoder_interface import MultimodalEncoder
from cline_version.model_meta import ModelMeta
import numpy as np

class MyCustomModel(MultimodalEncoder):
    def __init__(self):
        super().__init__()
        self.model_name = "my-custom-model"
        
        # Set model metadata
        self.mteb_model_meta = ModelMeta(
            name="my-custom-model",
            modalities=["text", "image"],
            framework="custom"
        )
    
    def encode_text(self, texts, *, task_name, **kwargs):
        # Your text encoding logic here
        return np.random.randn(len(texts), 384)
    
    def encode_image(self, images, *, task_name, **kwargs):
        # Your image encoding logic here
        return np.random.randn(len(images), 384)

# Use your custom model
model = MyCustomModel()
evaluator = cv.MultimodalMTEB(tasks=tasks)
results = evaluator.run(model, output_folder="results")
```

### Benchmark Evaluation

```python
# Load a predefined benchmark
benchmark = cv.get_benchmark("SAMPLE_MULTIMODAL_BENCHMARK")
print(f"Benchmark: {benchmark.name}")
print(f"Tasks: {len(benchmark.tasks)}")

# Run evaluation on the benchmark
evaluator = cv.MultimodalMTEB(tasks=benchmark.tasks)
results = evaluator.run(model, output_folder="results")
```

## 🔄 API Comparison with MTEB

The framework maintains API compatibility with MTEB patterns:

| MTEB | Cline Version | Purpose |
|------|---------------|---------|
| `mteb.get_tasks()` | `cv.get_tasks()` | Get filtered tasks |
| `mteb.get_task()` | `cv.get_task()` | Get specific task |
| `mteb.MTEB()` | `cv.MultimodalMTEB()` | Main evaluator |
| `mteb.get_model()` | `cv.get_model()` | Load model |

## 🎯 Key Differences from MTEB

1. **FAISS Integration**: Uses FAISS for efficient similarity search instead of computing full similarity matrices
2. **Multimodal Focus**: Specialized for text-image and multimodal retrieval tasks
3. **Performance Optimized**: Designed for your existing FAISS-based evaluation workflow
4. **Custom Metrics**: Optimized nDCG calculation using your preferred index-based approach

## 📊 Evaluation Metrics

The framework provides comprehensive retrieval metrics:

- **nDCG@100**: Normalized Discounted Cumulative Gain
- **Recall@k**: Recall at different cutoffs (1, 5, 10, 100)
- **MAP**: Mean Average Precision
- **Evaluation Time**: Performance benchmarking

## 🗂️ Dataset Format

Datasets follow a simple JSON Lines format:

**queries.jsonl**:
```json
{"id": "q1", "text": "A red car parked on the street"}
{"id": "q2", "text": "A cat sitting on a windowsill"}
```

**corpus.jsonl**:
```json
{"id": "doc1", "text": "Red sports car parked on city street", "image_path": "images/red_car.jpg"}
{"id": "doc2", "text": "Orange tabby cat sitting by window", "image_path": "images/cat_window.jpg"}
```

**qrels/test.tsv**:
```
q1	doc1	1
q1	doc21	1
q2	doc2	1
```

## 🚀 Performance Tips

1. **FAISS Index Selection**:
   - `IndexFlatIP`: Exact search (default)
   - `IndexIVFFlat`: Approximate search for large datasets
   - `IndexHNSW`: Fast approximate search

2. **Memory Optimization**:
   - Use appropriate batch sizes
   - Consider memory mapping for large datasets
   - Use FAISS-GPU for acceleration

3. **Evaluation Configuration**:
   ```python
   # Configure FAISS index type
   evaluator = cv.MultimodalMTEB(tasks=tasks)
   results = evaluator.run(
       model=model,
       faiss_index_type="IndexIVFFlat",  # For large datasets
       top_k=100
   )
   ```

## 📁 Sample Dataset

The framework includes a sample dataset with:
- 20 text queries
- 40 multimodal corpus items (text + image paths)
- Relevance judgments for evaluation
- Test split configuration

## 🔧 Extending the Framework

### Adding a New Task

```python
from cline_version.abstasks.AbsTaskMultimodalRetrieval import AbsTaskMultimodalRetrieval
from cline_version.abstasks.TaskMetadata import TaskMetadata

class MyNewTask(AbsTaskMultimodalRetrieval):
    metadata = TaskMetadata(
        name="MyNewTask",
        description="My custom retrieval task",
        type="Retrieval",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs=["eng"],
        main_score="ndcg@100"
    )
```

### Adding a New Model

```python
# Register your model in models/__init__.py
MODEL_REGISTRY["my-model"] = MyCustomModel
```

## 📁 Dónde se Guardan los Resultados

Los resultados se guardan automáticamente siguiendo la estructura de MTEB:

```
results/
└── {model_name}/
    └── {model_revision}/
        ├── model_meta.json              # Metadatos del modelo
        └── {task_name}.json             # Resultados de cada tarea
```

### Ejemplo de Estructura:
```
results/
└── sample-multimodal-encoder/
    └── v1.0/
        ├── model_meta.json
        └── SampleImageTextRetrieval.json
```

### Formato de Archivo de Resultados:
```json
{
  "task_name": "SampleImageTextRetrieval",
  "scores": {
    "test": [
      {
        "hf_subset": "default",
        "ndcg@100": 0.8234,
        "recall@1": 0.6500,
        "recall@5": 0.8000,
        "recall@10": 0.8500,
        "map": 0.7123
      }
    ]
  },
  "evaluation_time": 12.34,
  "dataset_revision": "main",
  "mteb_version": "0.1.0"
}
```

## 📈 Análisis de Resultados

### 1. Acceso Directo Durante Evaluación:
```python
# Los resultados se retornan directamente
results = evaluator.run(model, output_folder="results")

for result in results:
    print(f"Tarea: {result.task_name}")
    test_scores = result.scores["test"][0]
    print(f"nDCG@100: {test_scores['ndcg@100']}")
```

### 2. Cargar Resultados Guardados:
```python
from cline_version.evaluation.MultimodalMTEB import BenchmarkResults

# Cargar todos los resultados de una carpeta
results = BenchmarkResults("results")

# Print leaderboard
results.print_leaderboard()

# Get specific task performance
task_scores = results.get_leaderboard("SampleImageTextRetrieval")
print(task_scores)
```

### 3. Cargar Resultado Individual:
```python
from cline_version.evaluation.MultimodalMTEB import TaskResult

# Cargar resultado específico
result = TaskResult.from_disk("results/model-name/v1.0/TaskName.json")
print(result.scores)
```

### 4. Configurar Ubicación de Resultados:
```python
# Carpeta personalizada
results = evaluator.run(model, output_folder="mis_resultados")

# No guardar en disco
results = evaluator.run(model, output_folder=None)
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Add your task/model implementation
4. Update the registries
5. Add tests and documentation
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the [MTEB](https://github.com/embeddings-benchmark/mteb) framework
- Uses [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- Designed for multimodal retrieval evaluation workflows

---

**Ready to evaluate your multimodal retrieval models? Check out the [examples](examples/) to get started!**

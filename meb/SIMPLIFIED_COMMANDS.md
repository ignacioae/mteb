# meb Simplified Commands

El sistema meb ha sido simplificado para eliminar la capa de adapters, manteniendo solo la funcionalidad esencial de SLERP para fusión multimodal.

## Arquitectura Simplificada

```
Encoder (embeddings directos) → SLERP Fusion (α,β) → FAISS Search → NDCG
```

**Lo que se eliminó:**
- Capa de adapters (PCA, normalización, rotación, etc.)
- Transformaciones adicionales de embeddings

**Lo que se mantiene:**
- Fusión multimodal SLERP entre embeddings de texto e imagen
- Parámetros α (alpha) y β (beta) para controlar la fusión
- Motor FAISS con búsqueda eficiente
- Evaluación NDCG con matching por ID

## Comandos Bash Simplificados

### 1. Evaluación Básica con Embeddings Precomputados

```bash
# Evaluación estándar con dataset provider
python meb/run_evaluation.py --encoder precomputed --dataset provider

# Evaluación con dataset específico
python meb/run_evaluation.py --encoder precomputed --dataset fashion_ecommerce
python meb/run_evaluation.py --encoder precomputed --dataset electronics
```

### 2. Evaluación con Diferentes Encoders

```bash
# Encoder precomputado (por defecto)
python meb/run_evaluation.py --encoder precomputed --dataset provider

# Encoder mock para testing
python meb/run_evaluation.py --encoder mock --dataset provider

# Encoder HuggingFace (si está disponible)
python meb/run_evaluation.py --encoder sentence-transformers/all-MiniLM-L6-v2 --dataset provider

# Google API Encoder (requiere configuración de Google Cloud)
python meb/run_evaluation.py --encoder google_api --dataset provider
```

### 2.1. Google API Encoder

El nuevo **Google API Encoder** utiliza las APIs de Google Cloud Vertex AI para generar embeddings de alta calidad tanto para texto como para imágenes.

#### Configuración Requerida

```bash
# 1. Instalar dependencias
pip install -r meb/requirements_google_api.txt

# 2. Configurar autenticación de Google Cloud
export GOOGLE_CLOUD_PROJECT=tu-project-id
gcloud auth application-default login

# 3. Habilitar APIs necesarias
gcloud services enable aiplatform.googleapis.com
```

#### Uso Básico

```bash
# Configuración simple (usa variable de entorno GOOGLE_CLOUD_PROJECT)
export GOOGLE_CLOUD_PROJECT=tu-project-id
python meb/run_evaluation.py --encoder google_api --dataset provider

# Configuración avanzada con archivo JSON
python meb/run_evaluation.py --config google_api_config.json
```

#### Configuración Avanzada

Crear `google_api_config.json`:

```json
{
  "encoder": {
    "type": "google_api",
    "project_id": "tu-project-id",
    "location": "us-central1",
    "text_model": "text-embedding-004",
    "image_model": "multimodalembedding@001",
    "rate_limit_requests_per_minute": 60,
    "batch_size": 5,
    "max_retries": 3,
    "name": "google_api_custom"
  },
  "datasets": "provider"
}
```

#### Modelos Disponibles

- **Texto**: `text-embedding-004`, `text-multilingual-embedding-002`
- **Imágenes**: `multimodalembedding@001`
- **Dimensiones**: 768 (texto), 1408 (imagen)

#### Características

- ✅ **Rate limiting** automático para respetar límites de API
- ✅ **Batch processing** para eficiencia
- ✅ **Retry logic** con backoff exponencial
- ✅ **Caching** integrado para evitar llamadas duplicadas
- ✅ **Async processing** para mejor rendimiento
- ✅ **Error handling** robusto

#### Ejemplo de Uso Programático

```python
from meb.models.google_api_encoder import GoogleAPIEncoder

# Configuración básica
encoder = GoogleAPIEncoder(project_id="tu-project-id")

# Generar embeddings
text_embeddings = encoder.encode_text(["Texto de ejemplo"])
image_embeddings = encoder.encode_image(["imagen.jpg"])

# Ver información del modelo
print(encoder.get_model_info())
```

### 3. Control de Fusión Multimodal

Los parámetros α y β se evalúan automáticamente en el rango [0.0, 0.25, 0.5, 0.75, 1.0]:

- **α=0, β=0**: Solo texto
- **α=1, β=1**: Solo imagen  
- **α=0.5, β=0.5**: Fusión balanceada

```bash
# Evaluación completa con todos los parámetros α,β
python meb/run_evaluation.py --encoder precomputed --dataset provider

# Los resultados mostrarán el mejor α,β para cada dataset
```

### 4. Opciones Adicionales

```bash
# Usar GPU para FAISS (si está disponible)
python meb/run_evaluation.py --encoder precomputed --dataset provider --use-gpu

# Especificar directorio de salida
python meb/run_evaluation.py --encoder precomputed --dataset provider --output-dir my_results

# Usar archivo de configuración personalizado
python meb/run_evaluation.py --config my_config.json
```

### 5. Evaluación de Todos los Datasets

```bash
# Evaluar todos los datasets disponibles
python meb/run_evaluation.py --encoder precomputed

# Esto evaluará automáticamente: provider, fashion_ecommerce, electronics
```

## Resultados

Los resultados se guardan en:
- **JSON**: `results/advanced_evaluation_YYYYMMDD_HHMMSS.json`
- **Resumen**: `results/advanced_summary_YYYYMMDD_HHMMSS.txt`

### Ejemplo de Salida

```
🎯 ADVANCED EVALUATION COMPLETE!
==================================================
📁 Results saved to: results/advanced_evaluation_20250911_131452.json
📄 Summary saved to: results/advanced_summary_20250911_131452.txt

📊 Key Findings:
   • Encoder: precomputed
   • Datasets evaluated: 1
   • Average NDCG@5: 1.0000 ± 0.0000
   • Best parameters: α=0.0, β=0.0
```

## Configuración Personalizada

Crear un archivo `config.json`:

```json
{
  "encoder": "precomputed",
  "datasets": "provider",
  "alpha_values": [0.0, 0.5, 1.0],
  "beta_values": [0.0, 0.5, 1.0],
  "k_values": [1, 5, 10],
  "index_type": "flat",
  "engine": {
    "use_gpu": false,
    "cache_dir": "cache"
  }
}
```

Luego ejecutar:

```bash
python meb/run_evaluation.py --config config.json
```

## Ventajas del Sistema Simplificado

1. **Más directo**: Sin capas intermedias de transformación
2. **Más rápido**: Menos procesamiento de embeddings
3. **Más claro**: Flujo de evaluación más fácil de entender
4. **Mantiene SLERP**: Conserva la funcionalidad multimodal esencial
5. **Compatible**: Funciona con todos los datasets existentes

## Casos de Uso Típicos

```bash
# Evaluación rápida de un dataset
python meb/run_evaluation.py --encoder precomputed --dataset provider

# Evaluación completa de todos los datasets
python meb/run_evaluation.py --encoder precomputed

# Evaluación con GPU para datasets grandes
python meb/run_evaluation.py --encoder precomputed --use-gpu

# Evaluación con encoder personalizado
python meb/run_evaluation.py --encoder my_custom_encoder --dataset provider

# 🆕 BENCHMARKING COMPLETO: Todos los encoders x todos los datasets
python meb/run_evaluation.py --benchmark-all
```

## 🚀 Nuevo: Benchmarking Completo

### Comando `--benchmark-all`

El nuevo flag `--benchmark-all` ejecuta una evaluación comprehensiva de **todos los encoders disponibles** en **todos los datasets disponibles**, generando un análisis comparativo completo.

#### Uso Básico

```bash
# Benchmarking automático (detecta encoders disponibles)
python meb/run_evaluation.py --benchmark-all

# Con GPU para mayor velocidad
python meb/run_evaluation.py --benchmark-all --use-gpu

# Especificar directorio de salida
python meb/run_evaluation.py --benchmark-all --output-dir benchmark_results
```

#### Detección Automática de Encoders

El sistema detecta automáticamente qué encoders están disponibles:

- ✅ **`precomputed`**: Siempre disponible (baseline)
- ✅ **`google_adapter`**: Si existe `utils/models/adapter.onnx`
- ✅ **`google_api`**: Si `GOOGLE_CLOUD_PROJECT` está configurado
- ❌ **`mock`**: Excluido del benchmarking (solo desarrollo)

#### Salida del Benchmarking

```
🔍 Benchmark mode: Detecting available encoders...
✅ precomputed: Available (baseline)
✅ google_adapter: Available (ONNX model found)
❌ google_api: Not available (GOOGLE_CLOUD_PROJECT not set)

🚀 Starting comprehensive benchmark
📊 Encoders to test: ['precomputed', 'google_adapter']
📁 Datasets to evaluate: ['provider', 'fashion_ecommerce', 'electronics']

🔄 Testing encoder: precomputed
  📊 Evaluating dataset: provider
    ✅ NDCG@5: 1.0000 (α=0.0, β=0.0)
  📊 Evaluating dataset: fashion_ecommerce
    ✅ NDCG@5: 0.8500 (α=0.5, β=0.5)
  📊 Evaluating dataset: electronics
    ✅ NDCG@5: 0.9200 (α=0.25, β=0.75)

🔄 Testing encoder: google_adapter
  📊 Evaluating dataset: provider
    ✅ NDCG@5: 0.9800 (α=0.0, β=0.25)
  📊 Evaluating dataset: fashion_ecommerce
    ✅ NDCG@5: 0.8700 (α=0.5, β=0.5)
  📊 Evaluating dataset: electronics
    ✅ NDCG@5: 0.9100 (α=0.25, β=0.5)
```

#### Resultados Generados

El benchmarking genera dos archivos:

1. **`benchmark_all_YYYYMMDD_HHMMSS.json`**: Resultados completos en JSON
2. **`benchmark_summary_YYYYMMDD_HHMMSS.txt`**: Resumen legible

#### Ejemplo de Resumen

```
meb Comprehensive Benchmark Results

🏆 ENCODER RANKINGS:
1. precomputed: 0.9233 ± 0.0764 (3/3 datasets)
2. google_adapter: 0.9200 ± 0.0458 (3/3 datasets)

📊 DATASET DIFFICULTY:
electronics: 0.9150 ± 0.0071 (2/2 encoders)
provider: 0.9900 ± 0.0141 (2/2 encoders)
fashion_ecommerce: 0.8600 ± 0.0141 (2/2 encoders)

📋 DETAILED RESULTS MATRIX:
Encoder         | Provider | Fashion  | Electronics | Average
----------------------------------------------------------------
precomputed     | 1.0000   | 0.8500   | 0.9200      | 0.9233
google_adapter  | 0.9800   | 0.8700   | 0.9100      | 0.9200
```

#### Análisis Automático

El sistema calcula automáticamente:

- **Ranking de encoders** por NDCG@5 promedio
- **Dificultad de datasets** (menor NDCG@5 = más difícil)
- **Mejor encoder overall**
- **Dataset más difícil**
- **Matriz de resultados completa**
- **Estadísticas de éxito/fallo**

#### Ventajas del Benchmarking

1. **Comparación objetiva** entre todos los encoders disponibles
2. **Detección automática** de configuraciones disponibles
3. **Análisis estadístico** completo con medias y desviaciones
4. **Identificación de fortalezas** por dataset y encoder
5. **Resultados reproducibles** con timestamps y configuraciones guardadas

#### Tiempo de Ejecución

- **precomputed**: ~30 segundos por dataset (usa embeddings precalculados)
- **google_adapter**: ~2-5 minutos por dataset (transformaciones ONNX)
- **google_api**: ~5-15 minutos por dataset (llamadas API + rate limiting)

**Total estimado**: 15-60 minutos dependiendo de encoders disponibles y datasets.

## 🆕 Nueva Funcionalidad: Embeddings Precomputados

El `Adapter` ahora soporta trabajar directamente con embeddings precomputados, además del método tradicional desde textos/imágenes.

### Métodos Disponibles

#### 1. Método Tradicional (desde textos/imágenes)
```python
from models.adapter import Adapter

finetuned = Adapter(base_encoder="mock")
text_embeddings = finetuned.encode_text(["texto 1", "texto 2"])
image_embeddings = finetuned.encode_image(["img1.jpg", "img2.jpg"])
```

#### 2. Nuevo Método (desde embeddings precomputados)
```python
from models.finetuning_adapter import Adapter
from models.encoders import MockEncoder

# Obtener embeddings base de cualquier encoder
base_encoder = MockEncoder()
base_text_emb = base_encoder.encode_text(["texto 1", "texto 2"])
base_image_emb = base_encoder.encode_image(["img1.jpg", "img2.jpg"])

# Aplicar transformaciones de fine-tuning
finetuned = Adapter()
transformed_text = finetuned.transform_text_embeddings(base_text_emb)
transformed_image = finetuned.transform_image_embeddings(base_image_emb)
```

#### 3. Transformación a Nivel de Dataset (con caché)
```python
# Transformar dataset completo con soporte de caché
text_emb, image_emb = finetuned.transform_embeddings_dataset(
    base_text_emb, base_image_emb, 
    dataset_name="mi_dataset", 
    data_type="catalog",
    use_cache=True
)
```

### Ventajas de los Embeddings Precomputados

1. **Eficiencia**: No necesitas recomputar embeddings base si ya los tienes
2. **Flexibilidad**: Puedes usar cualquier encoder base y aplicar fine-tuning después
3. **Compatibilidad**: Mantiene compatibilidad total con el método tradicional
4. **Caché**: Soporte completo de caché para transformaciones

### Casos de Uso

```python
# Caso 1: Tienes embeddings de un modelo costoso y quieres aplicar fine-tuning
expensive_embeddings = load_expensive_model_embeddings()
finetuned_embeddings = finetuned_encoder.transform_text_embeddings(expensive_embeddings)

# Caso 2: Experimentar con diferentes transformaciones sobre los mismos embeddings base
base_embeddings = base_encoder.encode_text(texts)
linear_transform = Adapter(transformation_type="linear")
mlp_transform = Adapter(transformation_type="mlp")

result1 = linear_transform.transform_text_embeddings(base_embeddings)
result2 = mlp_transform.transform_text_embeddings(base_embeddings)

# Caso 3: Pipeline de procesamiento por lotes
for batch in large_dataset:
    base_emb = batch_encoder.encode_text(batch.texts)
    transformed_emb = finetuned.transform_text_embeddings(base_emb)
    save_to_database(transformed_emb)
```

### Pruebas

Para probar la nueva funcionalidad:

```bash
# Ejecutar script de pruebas
cd meb && python test_precomputed_embeddings.py
```

Este script verifica que:
- Los métodos tradicionales y nuevos dan resultados consistentes
- El caché funciona correctamente
- Las transformaciones se aplican correctamente

from __future__ import annotations

from meb.benchmarks.sample_benchmark import SAMPLE_MULTIMODAL_BENCHMARK
from meb.evaluation import *
from meb.encoder_interface import Encoder, MultimodalEncoder, AdapterEncoder
from meb.models import get_model, get_model_meta, get_model_metas
from meb.overview import (
    TASKS_REGISTRY, 
    get_task, 
    get_tasks, 
    list_available_tasks,
    print_available_tasks
)

from .benchmarks.sample_benchmark import SampleBenchmark
from .benchmarks.get_benchmark import BENCHMARK_REGISTRY, get_benchmark, get_benchmarks

__version__ = "0.1.0"

__all__ = [
    "SAMPLE_MULTIMODAL_BENCHMARK",
    "TASKS_REGISTRY",
    "get_tasks",
    "get_task",
    "list_available_tasks",
    "print_available_tasks",
    "get_model",
    "get_model_meta", 
    "get_model_metas",
    "Encoder",
    "MultimodalEncoder",
    "AdapterEncoder",
    "SampleBenchmark",
    "get_benchmark",
    "get_benchmarks",
    "BENCHMARK_REGISTRY",
    "MultimodalMTEB",
]

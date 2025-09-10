from __future__ import annotations

from mtbe.overview import get_tasks


class SampleBenchmark:
    """Sample benchmark for multimodal retrieval tasks."""
    
    def __init__(self):
        self.name = "SAMPLE_MULTIMODAL_BENCHMARK"
        self.description = "A sample benchmark for testing multimodal retrieval capabilities"
        self.tasks = get_tasks(tasks=["SampleImageTextRetrieval"])
        self.citation = """
        @misc{sample_multimodal_benchmark_2024,
            title={Sample Multimodal Benchmark for Text-Image Retrieval},
            author={Cline Framework},
            year={2024},
            note={Sample benchmark for demonstration purposes}
        }
        """
    
    def __repr__(self):
        return f"SampleBenchmark(name='{self.name}', tasks={len(self.tasks)})"


# Create the benchmark instance
SAMPLE_MULTIMODAL_BENCHMARK = SampleBenchmark()

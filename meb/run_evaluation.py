import sys
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.encoders import get_encoder, PrecomputedEncoder
from utils.faiss_engine import FAISSEvaluationEngine
from utils.dataset_utils import load_dataset, list_available_datasets

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultimodalEmbeddingsBenchmark:
    """
    Evaluation pipeline for multimodal embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluation pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.encoder = None
        self.engine = None
        
        # Initialize components
        self._initialize_encoder()
        self._initialize_engine()
        
        logger.info("Initialized advanced evaluation pipeline")
    
    def _initialize_encoder(self):
        """Initialize the encoder from configuration."""
        encoder_config = self.config.get("encoder", "precomputed")
        self.encoder = get_encoder(encoder_config)
        logger.info(f"Initialized encoder: {self.encoder.name}")
    
    
    def _initialize_engine(self):
        """Initialize the FAISS evaluation engine."""
        engine_config = self.config.get("engine", {})
        self.engine = FAISSEvaluationEngine(
            use_gpu=engine_config.get("use_gpu", False),
            cache_dir=engine_config.get("cache_dir", "cache")
        )
        logger.info("Initialized FAISS evaluation engine")
    
    def load_or_generate_embeddings(
        self, dataset_name: str, data_type: str, 
        texts: List[str], image_paths: List[str],
        use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load or generate embeddings for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            data_type: Type of data ('catalog' or 'test')
            texts: List of text strings
            image_paths: List of image paths
            use_cache: Whether to use cached embeddings
            
        Returns:
            Tuple of (text_embeddings, image_embeddings)
        """
        # For precomputed encoder, we need to handle differently
        if isinstance(self.encoder, PrecomputedEncoder):
            # This should be handled by the calling function
            raise ValueError("PrecomputedEncoder requires special handling")
        
        # Generate embeddings using the encoder (no adapter transformations)
        text_embeddings, image_embeddings = self.encoder.encode_dataset(
            texts, image_paths, dataset_name, data_type, use_cache
        )
        
        return text_embeddings, image_embeddings
    
    def evaluate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Evaluate a single dataset.
        
        Args:
            dataset_name: Name of the dataset to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating dataset: {dataset_name}")
        
        # Load dataset metadata
        try:
            if isinstance(self.encoder, PrecomputedEncoder):
                # Load precomputed embeddings from CSV
                catalog_data, test_data = load_dataset(dataset_name)
                
                # Unpack data (no adapter transformations for simplified pipeline)
                catalog_ids, catalog_texts, catalog_image_paths, catalog_metadata, catalog_text_embeddings, catalog_image_embeddings = catalog_data
                test_ids, test_texts, test_image_paths, test_metadata, test_product_counts, test_text_embeddings, test_image_embeddings = test_data
            
            else:
                # Load raw data and generate embeddings
                catalog_data, test_data = load_dataset(dataset_name)
                
                # Unpack data (without embeddings)
                catalog_ids, catalog_texts, catalog_image_paths, catalog_metadata, _, _ = catalog_data
                test_ids, test_texts, test_image_paths, test_metadata, test_product_counts, _, _ = test_data
                
                # Generate embeddings
                catalog_text_embeddings, catalog_image_embeddings = self.load_or_generate_embeddings(
                    dataset_name, "catalog", catalog_texts, catalog_image_paths
                )
                test_text_embeddings, test_image_embeddings = self.load_or_generate_embeddings(
                    dataset_name, "test", test_texts, test_image_paths
                )
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return {"error": str(e)}
        
        # Get evaluation parameters
        alpha_values = self.config.get("alpha_values", [0.0, 0.25, 0.5, 0.75, 1.0])
        beta_values = self.config.get("beta_values", [0.0, 0.25, 0.5, 0.75, 1.0])
        k_values = self.config.get("k_values", [1, 5, 10, 50])
        index_type = self.config.get("index_type", "flat")
        
        logger.info(f"Parameters: α={alpha_values}, β={beta_values}, k={k_values}")
        
        # Store catalog metadata in the engine for ID matching
        self.engine.set_catalog_metadata(catalog_ids, catalog_image_paths, catalog_texts)
        
        # Pre-compute FAISS indices for all beta values
        self.engine.precompute_catalog_indices(
            catalog_text_embeddings, catalog_image_embeddings, beta_values, index_type
        )
        
        # Evaluate all queries in batch
        retrieval_results = self.engine.evaluate_query_batch(
            test_text_embeddings, test_image_embeddings, 
            alpha_values, beta_values, max(k_values)
        )
        
        # Calculate NDCG metrics for all combinations
        evaluation_results = {}
        
        for combo_key, retrieval_result in retrieval_results.items():
            alpha = retrieval_result["alpha"]
            beta = retrieval_result["beta"]
            indices = retrieval_result["indices"]  # (n_queries, k)
            
            # Create relevance scores based on ID matching (multi-instance support)
            relevance_scores = []
            id_match_stats = []  # Track matching statistics
            
            for i in range(len(test_ids)):
                query_id = test_ids[i]
                total_instances = self.engine.id_counts.get(query_id, 0)
                
                # Assign relevance 1.0 to all catalog items matching the query ID
                query_relevance = []
                matches_found = 0
                for j, catalog_id in enumerate(catalog_ids):
                    if query_id == catalog_id:
                        query_relevance.append(1.0)  # Perfect relevance for ID match
                        matches_found += 1
                    else:
                        query_relevance.append(0.0)  # No relevance for non-match
                
                relevance_scores.append(query_relevance)
                id_match_stats.append({
                    "query_id": query_id,
                    "total_instances": total_instances,
                    "matches_in_catalog": matches_found
                })
            
            logger.info(f"ID matching: {len([s for s in id_match_stats if s['total_instances'] > 0])} queries have catalog matches")
            
            # Calculate NDCG for all k values
            ndcg_results = self.engine.calculate_ndcg_batch(indices, relevance_scores, k_values)
            
            # Aggregate results
            aggregated_metrics = {}
            for k in k_values:
                ndcg_scores = ndcg_results[f"ndcg@{k}"]
                aggregated_metrics[f"ndcg@{k}"] = {
                    "mean": float(np.mean(ndcg_scores)),
                    "std": float(np.std(ndcg_scores)),
                    "min": float(np.min(ndcg_scores)),
                    "max": float(np.max(ndcg_scores))
                }
            
            evaluation_results[combo_key] = {
                "alpha": alpha,
                "beta": beta,
                "metrics": aggregated_metrics,
                "n_queries": len(test_ids)
            }
        
        # Clean up FAISS indices to free memory
        self.engine.cleanup_indices()
        
        # Find best parameters
        best_combo = None
        best_ndcg = 0.0
        
        for combo_key, result in evaluation_results.items():
            ndcg_5 = result["metrics"]["ndcg@5"]["mean"]
            if ndcg_5 > best_ndcg:
                best_ndcg = ndcg_5
                best_combo = (result["alpha"], result["beta"])
        
        dataset_result = {
            "dataset_name": dataset_name,
            "encoder": self.encoder.name,
            "catalog_size": len(catalog_ids),
            "test_size": len(test_ids),
            "parameter_evaluations": evaluation_results,
            "best_parameters": {
                "alpha": best_combo[0] if best_combo else 0.5,
                "beta": best_combo[1] if best_combo else 0.5,
                "ndcg@5": best_ndcg
            }
        }
        
        logger.info(f"Best parameters for {dataset_name}: α={best_combo[0]}, β={best_combo[1]} (NDCG@5: {best_ndcg:.4f})")
        
        return dataset_result
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation on all configured datasets.
        
        Returns:
            Complete evaluation results
        """
        logger.info("Starting advanced evaluation")
        
        # Get datasets to evaluate
        datasets_config = self.config.get("datasets", "all")
        
        if datasets_config == "all":
            available_datasets = list_available_datasets()
            dataset_names = list(available_datasets.keys())
        elif isinstance(datasets_config, list):
            dataset_names = datasets_config
        else:
            dataset_names = [datasets_config]
        
        logger.info(f"Evaluating {len(dataset_names)} datasets: {dataset_names}")
        
        # Initialize results
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "encoder": self.encoder.name,
            "datasets_evaluated": [],
            "summary": {}
        }
        
        # Evaluate each dataset
        for dataset_name in dataset_names:
            try:
                dataset_result = self.evaluate_dataset(dataset_name)
                all_results["datasets_evaluated"].append(dataset_result)
            except Exception as e:
                logger.error(f"Failed to evaluate dataset {dataset_name}: {e}")
                continue
        
        # Calculate summary statistics
        if all_results["datasets_evaluated"]:
            all_ndcg_5_scores = []
            best_params_list = []
            
            for dataset_result in all_results["datasets_evaluated"]:
                if "error" not in dataset_result:
                    best_params = dataset_result["best_parameters"]
                    all_ndcg_5_scores.append(best_params["ndcg@5"])
                    best_params_list.append((best_params["alpha"], best_params["beta"]))
            
            # Find most common best parameters
            from collections import Counter
            param_counter = Counter(best_params_list)
            most_common_params = param_counter.most_common(1)[0][0] if param_counter else (0.5, 0.5)
            
            all_results["summary"] = {
                "total_datasets": len(all_results["datasets_evaluated"]),
                "successful_evaluations": len([d for d in all_results["datasets_evaluated"] if "error" not in d]),
                "average_ndcg_5": float(np.mean(all_ndcg_5_scores)) if all_ndcg_5_scores else 0.0,
                "std_ndcg_5": float(np.std(all_ndcg_5_scores)) if all_ndcg_5_scores else 0.0,
                "most_common_best_params": {
                    "alpha": most_common_params[0],
                    "beta": most_common_params[1]
                },
                "parameter_distribution": {f"alpha_{params[0]}_beta_{params[1]}": count for params, count in param_counter.items()}
            }
        
        return all_results


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Return default configuration
        return {
            "encoder": "precomputed",
            "datasets": "all",
            "alpha_values": [0.0, 0.25, 0.5, 0.75, 1.0],
            "beta_values": [0.0, 0.25, 0.5, 0.75, 1.0],
            "k_values": [1, 5, 10, 50],
            "index_type": "flat",
            "engine": {
                "use_gpu": False,
                "cache_dir": "cache"
            }
        }


def save_results(results: Dict[str, Any], output_dir: str = "advanced_results"):
    """
    Save evaluation results to files.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON results
    json_file = os.path.join(output_dir, f"advanced_evaluation_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save text summary
    summary_file = os.path.join(output_dir, f"advanced_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("meb Advanced Evaluation System Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Evaluation Date: {results['timestamp']}\n")
        f.write(f"Encoder: {results['encoder']}\n")
        f.write(f"Datasets Evaluated: {results['summary']['total_datasets']}\n")
        f.write(f"Successful Evaluations: {results['summary']['successful_evaluations']}\n")
        f.write(f"Average NDCG@5: {results['summary']['average_ndcg_5']:.4f} ± {results['summary']['std_ndcg_5']:.4f}\n")
        f.write(f"Most Common Best Parameters:\n")
        f.write(f"  - Alpha: {results['summary']['most_common_best_params']['alpha']}\n")
        f.write(f"  - Beta: {results['summary']['most_common_best_params']['beta']}\n")
        f.write(f"Parameter Distribution: {results['summary']['parameter_distribution']}\n\n")
        
        # Detailed results per dataset
        for dataset_result in results["datasets_evaluated"]:
            if "error" in dataset_result:
                f.write(f"\nDataset: {dataset_result.get('dataset_name', 'Unknown')} - ERROR: {dataset_result['error']}\n")
                continue
            
            f.write(f"\nDataset: {dataset_result['dataset_name']}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Catalog Size: {dataset_result['catalog_size']}\n")
            f.write(f"Test Queries: {dataset_result['test_size']}\n")
            f.write(f"Best Parameters: α={dataset_result['best_parameters']['alpha']}, β={dataset_result['best_parameters']['beta']}\n")
            f.write(f"Best NDCG@5: {dataset_result['best_parameters']['ndcg@5']:.4f}\n")
    
    logger.info(f"Results saved to {json_file} and {summary_file}")
    return json_file, summary_file


def detect_available_encoders() -> List[str]:
    """
    Automatically detect which encoders are available for benchmarking.
    
    Returns:
        List of available encoder names
    """
    available_encoders = []
    
    # precomputed is always available (baseline)
    available_encoders.append("precomputed")
    logger.info("✅ precomputed: Available (baseline)")
    
    # Check google_adapter (requires ONNX model)
    adapter_model_path = os.path.join(current_dir, "utils", "models", "adapter.onnx")
    if os.path.exists(adapter_model_path):
        available_encoders.append("google_adapter")
        logger.info("✅ google_adapter: Available (ONNX model found)")
    else:
        logger.info("❌ google_adapter: Not available (ONNX model not found)")
    
    # Check google_api (requires Google Cloud configuration)
    try:
        project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
        if project_id:
            # Try to import Google Cloud libraries
            try:
                import vertexai
                from google.auth import default
                available_encoders.append("google_api")
                logger.info(f"✅ google_api: Available (project: {project_id})")
            except ImportError:
                logger.info("❌ google_api: Not available (Google Cloud libraries not installed)")
        else:
            logger.info("❌ google_api: Not available (GOOGLE_CLOUD_PROJECT not set)")
    except Exception as e:
        logger.info(f"❌ google_api: Not available ({e})")
    
    return available_encoders


def run_benchmark_all(available_encoders: List[str], datasets: List[str], output_dir: str, use_gpu: bool = False) -> Dict[str, Any]:
    """
    Run benchmarking across all available encoders and datasets.
    
    Args:
        available_encoders: List of encoder names to test
        datasets: List of dataset names to evaluate
        output_dir: Output directory for results
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Comprehensive benchmark results
    """
    logger.info(f"🚀 Starting comprehensive benchmark")
    logger.info(f"📊 Encoders to test: {available_encoders}")
    logger.info(f"📁 Datasets to evaluate: {datasets}")
    
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_type": "comprehensive",
        "encoders_tested": available_encoders,
        "datasets_evaluated": datasets,
        "results": {},
        "summary": {}
    }
    
    # Results matrix: encoder -> dataset -> metrics
    results_matrix = {}
    
    for encoder_name in available_encoders:
        logger.info(f"\n🔄 Testing encoder: {encoder_name}")
        results_matrix[encoder_name] = {}
        
        for dataset_name in datasets:
            logger.info(f"  📊 Evaluating dataset: {dataset_name}")
            
            try:
                # Create configuration for this encoder/dataset combination
                config = {
                    "encoder": encoder_name,
                    "datasets": dataset_name,
                    "alpha_values": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "beta_values": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "k_values": [1, 5, 10, 50],
                    "index_type": "flat",
                    "engine": {
                        "use_gpu": use_gpu,
                        "cache_dir": "cache"
                    }
                }
                
                # Run evaluation
                pipeline = MultimodalEmbeddingsBenchmark(config)
                result = pipeline.run_evaluation()
                
                # Extract key metrics
                if result["datasets_evaluated"]:
                    dataset_result = result["datasets_evaluated"][0]
                    if "error" not in dataset_result:
                        best_params = dataset_result["best_parameters"]
                        results_matrix[encoder_name][dataset_name] = {
                            "ndcg@5": best_params["ndcg@5"],
                            "best_alpha": best_params["alpha"],
                            "best_beta": best_params["beta"],
                            "catalog_size": dataset_result["catalog_size"],
                            "test_size": dataset_result["test_size"]
                        }
                        logger.info(f"    ✅ NDCG@5: {best_params['ndcg@5']:.4f} (α={best_params['alpha']}, β={best_params['beta']})")
                    else:
                        results_matrix[encoder_name][dataset_name] = {"error": dataset_result["error"]}
                        logger.error(f"    ❌ Error: {dataset_result['error']}")
                else:
                    results_matrix[encoder_name][dataset_name] = {"error": "No results generated"}
                    logger.error(f"    ❌ No results generated")
                    
            except Exception as e:
                logger.error(f"    ❌ Failed to evaluate {encoder_name} on {dataset_name}: {e}")
                results_matrix[encoder_name][dataset_name] = {"error": str(e)}
    
    benchmark_results["results"] = results_matrix
    
    # Calculate summary statistics
    logger.info(f"\n📊 Calculating benchmark summary...")
    
    # Calculate average NDCG@5 per encoder
    encoder_averages = {}
    for encoder_name in available_encoders:
        ndcg_scores = []
        for dataset_name in datasets:
            result = results_matrix[encoder_name].get(dataset_name, {})
            if "ndcg@5" in result:
                ndcg_scores.append(result["ndcg@5"])
        
        if ndcg_scores:
            encoder_averages[encoder_name] = {
                "mean_ndcg@5": float(np.mean(ndcg_scores)),
                "std_ndcg@5": float(np.std(ndcg_scores)),
                "datasets_successful": len(ndcg_scores),
                "datasets_total": len(datasets)
            }
        else:
            encoder_averages[encoder_name] = {
                "mean_ndcg@5": 0.0,
                "std_ndcg@5": 0.0,
                "datasets_successful": 0,
                "datasets_total": len(datasets)
            }
    
    # Calculate average NDCG@5 per dataset
    dataset_averages = {}
    for dataset_name in datasets:
        ndcg_scores = []
        for encoder_name in available_encoders:
            result = results_matrix[encoder_name].get(dataset_name, {})
            if "ndcg@5" in result:
                ndcg_scores.append(result["ndcg@5"])
        
        if ndcg_scores:
            dataset_averages[dataset_name] = {
                "mean_ndcg@5": float(np.mean(ndcg_scores)),
                "std_ndcg@5": float(np.std(ndcg_scores)),
                "encoders_successful": len(ndcg_scores),
                "encoders_total": len(available_encoders)
            }
        else:
            dataset_averages[dataset_name] = {
                "mean_ndcg@5": 0.0,
                "std_ndcg@5": 0.0,
                "encoders_successful": 0,
                "encoders_total": len(available_encoders)
            }
    
    # Find best encoder overall
    best_encoder = max(encoder_averages.keys(), key=lambda x: encoder_averages[x]["mean_ndcg@5"])
    
    # Find hardest dataset (lowest average NDCG@5)
    hardest_dataset = min(dataset_averages.keys(), key=lambda x: dataset_averages[x]["mean_ndcg@5"])
    
    benchmark_results["summary"] = {
        "encoder_rankings": sorted(encoder_averages.items(), key=lambda x: x[1]["mean_ndcg@5"], reverse=True),
        "dataset_difficulty": sorted(dataset_averages.items(), key=lambda x: x[1]["mean_ndcg@5"]),
        "best_encoder": {
            "name": best_encoder,
            "mean_ndcg@5": encoder_averages[best_encoder]["mean_ndcg@5"]
        },
        "hardest_dataset": {
            "name": hardest_dataset,
            "mean_ndcg@5": dataset_averages[hardest_dataset]["mean_ndcg@5"]
        },
        "total_evaluations": len(available_encoders) * len(datasets),
        "successful_evaluations": sum(
            1 for encoder in results_matrix.values() 
            for result in encoder.values() 
            if "ndcg@5" in result
        )
    }
    
    return benchmark_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="meb Advanced Evaluation System")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--encoder", type=str, help="Encoder configuration (overrides config file)")
    parser.add_argument("--dataset", type=str, help="Single dataset to evaluate")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration for FAISS")
    parser.add_argument("--benchmark-all", action="store_true", help="Run comprehensive benchmark across all available encoders and datasets")
    
    args = parser.parse_args()
    
    # Handle benchmark-all mode
    if args.benchmark_all:
        logger.info("🔍 Benchmark mode: Detecting available encoders...")
        
        # Detect available encoders
        available_encoders = detect_available_encoders()
        
        if not available_encoders:
            logger.error("❌ No encoders available for benchmarking")
            sys.exit(1)
        
        # Get available datasets
        available_datasets = list_available_datasets()
        dataset_names = list(available_datasets.keys())
        
        if not dataset_names:
            logger.error("❌ No datasets available for benchmarking")
            sys.exit(1)
        
        logger.info(f"📊 Found {len(available_encoders)} encoders and {len(dataset_names)} datasets")
        
        # Run comprehensive benchmark
        try:
            benchmark_results = run_benchmark_all(
                available_encoders, 
                dataset_names, 
                args.output_dir, 
                args.use_gpu
            )
            
            # Save benchmark results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            benchmark_file = os.path.join(args.output_dir, f"benchmark_all_{timestamp}.json")
            benchmark_summary = os.path.join(args.output_dir, f"benchmark_summary_{timestamp}.txt")
            
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save JSON results
            with open(benchmark_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            
            # Save text summary
            with open(benchmark_summary, 'w') as f:
                f.write("meb Comprehensive Benchmark Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Benchmark Date: {benchmark_results['timestamp']}\n")
                f.write(f"Encoders Tested: {len(benchmark_results['encoders_tested'])}\n")
                f.write(f"Datasets Evaluated: {len(benchmark_results['datasets_evaluated'])}\n")
                f.write(f"Total Evaluations: {benchmark_results['summary']['total_evaluations']}\n")
                f.write(f"Successful Evaluations: {benchmark_results['summary']['successful_evaluations']}\n\n")
                
                # Encoder rankings
                f.write("🏆 ENCODER RANKINGS:\n")
                f.write("-" * 30 + "\n")
                for i, (encoder_name, stats) in enumerate(benchmark_results['summary']['encoder_rankings'], 1):
                    f.write(f"{i}. {encoder_name}: {stats['mean_ndcg@5']:.4f} ± {stats['std_ndcg@5']:.4f} "
                           f"({stats['datasets_successful']}/{stats['datasets_total']} datasets)\n")
                
                # Dataset difficulty
                f.write(f"\n📊 DATASET DIFFICULTY:\n")
                f.write("-" * 30 + "\n")
                for dataset_name, stats in benchmark_results['summary']['dataset_difficulty']:
                    f.write(f"{dataset_name}: {stats['mean_ndcg@5']:.4f} ± {stats['std_ndcg@5']:.4f} "
                           f"({stats['encoders_successful']}/{stats['encoders_total']} encoders)\n")
                
                # Results matrix
                f.write(f"\n📋 DETAILED RESULTS MATRIX:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Encoder':<15} | {'Provider':<8} | {'Fashion':<8} | {'Electronics':<8} | {'Average':<8}\n")
                f.write("-" * 70 + "\n")
                
                for encoder_name in benchmark_results['encoders_tested']:
                    encoder_results = benchmark_results['results'][encoder_name]
                    row_values = []
                    
                    for dataset_name in benchmark_results['datasets_evaluated']:
                        result = encoder_results.get(dataset_name, {})
                        if "ndcg@5" in result:
                            row_values.append(f"{result['ndcg@5']:.4f}")
                        else:
                            row_values.append("ERROR")
                    
                    # Calculate average for this encoder
                    valid_scores = [float(v) for v in row_values if v != "ERROR"]
                    avg_score = np.mean(valid_scores) if valid_scores else 0.0
                    
                    f.write(f"{encoder_name:<15} | {row_values[0] if len(row_values) > 0 else 'N/A':<8} | "
                           f"{row_values[1] if len(row_values) > 1 else 'N/A':<8} | "
                           f"{row_values[2] if len(row_values) > 2 else 'N/A':<8} | {avg_score:.4f}\n")
            
            # Print summary to console
            print(f"\n🎯 COMPREHENSIVE BENCHMARK COMPLETE!")
            print("=" * 60)
            print(f"📁 Results saved to: {benchmark_file}")
            print(f"📄 Summary saved to: {benchmark_summary}")
            print(f"\n📊 Key Findings:")
            print(f"   • Encoders tested: {len(benchmark_results['encoders_tested'])}")
            print(f"   • Datasets evaluated: {len(benchmark_results['datasets_evaluated'])}")
            print(f"   • Total evaluations: {benchmark_results['summary']['total_evaluations']}")
            print(f"   • Successful evaluations: {benchmark_results['summary']['successful_evaluations']}")
            print(f"   • Best encoder: {benchmark_results['summary']['best_encoder']['name']} "
                  f"(NDCG@5: {benchmark_results['summary']['best_encoder']['mean_ndcg@5']:.4f})")
            print(f"   • Hardest dataset: {benchmark_results['summary']['hardest_dataset']['name']} "
                  f"(NDCG@5: {benchmark_results['summary']['hardest_dataset']['mean_ndcg@5']:.4f})")
            
            print(f"\n🏆 Encoder Rankings:")
            for i, (encoder_name, stats) in enumerate(benchmark_results['summary']['encoder_rankings'], 1):
                print(f"   {i}. {encoder_name}: {stats['mean_ndcg@5']:.4f} ± {stats['std_ndcg@5']:.4f}")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # Standard single evaluation mode
        # Load configuration
        config = load_config(args.config)
        
        # Override with command line arguments
        if args.encoder:
            config["encoder"] = args.encoder
        if args.dataset:
            config["datasets"] = args.dataset
        if args.use_gpu:
            config["engine"]["use_gpu"] = True
        
        logger.info(f"Configuration: {config}")
        
        try:
            # Initialize and run evaluation
            pipeline = MultimodalEmbeddingsBenchmark(config)
            results = pipeline.run_evaluation()
            
            # Save results
            json_file, summary_file = save_results(results, args.output_dir)
            
            # Print summary
            print(f"\n🎯 ADVANCED EVALUATION COMPLETE!")
            print("=" * 50)
            print(f"📁 Results saved to: {json_file}")
            print(f"📄 Summary saved to: {summary_file}")
            print(f"\n📊 Key Findings:")
            print(f"   • Encoder: {results['encoder']}")
            print(f"   • Datasets evaluated: {results['summary']['total_datasets']}")
            print(f"   • Average NDCG@5: {results['summary']['average_ndcg_5']:.4f} ± {results['summary']['std_ndcg_5']:.4f}")
            print(f"   • Best parameters: α={results['summary']['most_common_best_params']['alpha']}, " +
                f"β={results['summary']['most_common_best_params']['beta']}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()

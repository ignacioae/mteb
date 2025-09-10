"""
Comprehensive demo of the CatalogInterpolationAdapter using CSV data files.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path to import mteb modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mteb.models.adapter_models import CatalogInterpolationAdapter, load_catalog_data, load_test_queries


def load_catalog2_data():
    """Load catalog2 data from CSV file."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'catalog2', 'catalog_data.csv')
    return load_catalog_data(csv_path)


def load_test2_data():
    """Load test2 data from CSV file."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'test2', 'test_queries.csv')
    return load_test_queries(csv_path)


def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def run_adapter_evaluation():
    """Run comprehensive adapter evaluation."""
    print("=== Comprehensive CatalogInterpolationAdapter Evaluation ===\n")
    
    # Load data
    print("Loading data...")
    skus, names, descriptions, categories, prices, stock_counts, catalog_embeddings = load_catalog2_data()
    queries, relevant_skus_list, product_counts_list, query_embeddings = load_test2_data()
    
    print(f"✓ Loaded {len(skus)} catalog items")
    print(f"✓ Loaded {len(queries)} test queries")
    print()
    
    # Display catalog information
    print("--- Catalog Items ---")
    for i, (sku, name, category, price, stock) in enumerate(zip(skus, names, categories, prices, stock_counts)):
        print(f"{sku}: {name} ({category}) - ${price:.2f} - Stock: {stock}")
    print()
    
    # Test different beta values
    beta_values = [0.0, 0.3, 0.7, 1.0]
    
    for beta in beta_values:
        print(f"=== Evaluation with Beta = {beta} ===")
        
        # Create adapter
        adapter = CatalogInterpolationAdapter(catalog_embeddings, beta=beta)
        
        total_enhanced_ndcg = 0.0
        total_standard_ndcg = 0.0
        
        for i, query in enumerate(queries):
            print(f"\nQuery {i+1}: '{query}'")
            
            # Get original query embedding
            original_query_embedding = query_embeddings[i]
            
            # Enhance query embedding
            enhanced_query_embedding = adapter.enhance_query_embedding(original_query_embedding)
            
            # Calculate similarities with catalog items
            original_similarities = []
            enhanced_similarities = []
            
            for j, catalog_embedding in enumerate(catalog_embeddings):
                orig_sim = calculate_cosine_similarity(original_query_embedding, catalog_embedding)
                enh_sim = calculate_cosine_similarity(enhanced_query_embedding, catalog_embedding)
                original_similarities.append((orig_sim, j))
                enhanced_similarities.append((enh_sim, j))
            
            # Sort by similarity (descending)
            original_similarities.sort(reverse=True)
            enhanced_similarities.sort(reverse=True)
            
            # Get relevant items for this query
            relevant_skus = relevant_skus_list[i]
            relevant_indices = [skus.index(sku) for sku in relevant_skus if sku in skus]
            
            print(f"  Relevant items: {relevant_skus}")
            
            # Show top 3 results for original vs enhanced
            print("  Original ranking (top 3):")
            for rank, (sim, idx) in enumerate(original_similarities[:3]):
                relevance = 1.0 if idx in relevant_indices else 0.0
                print(f"    {rank+1}. {skus[idx]} ({names[idx]}) - Sim: {sim:.3f}, Rel: {relevance}")
            
            print("  Enhanced ranking (top 3):")
            for rank, (sim, idx) in enumerate(enhanced_similarities[:3]):
                relevance = 1.0 if idx in relevant_indices else 0.0
                print(f"    {rank+1}. {skus[idx]} ({names[idx]}) - Sim: {sim:.3f}, Rel: {relevance}")
            
            # Calculate NDCG scores
            if relevant_indices:
                # Create relevance scores for NDCG calculation
                relevance_scores = []
                item_stock_counts = []
                
                for _, idx in enhanced_similarities:
                    relevance = 1.0 if idx in relevant_indices else 0.0
                    relevance_scores.append(relevance)
                    item_stock_counts.append(stock_counts[idx])
                
                # Calculate standard NDCG (equal weights)
                standard_weights = [1] * len(relevance_scores)
                standard_ndcg = adapter.calculate_enhanced_ndcg(relevance_scores, standard_weights, k=3)
                
                # Calculate enhanced NDCG (with stock weighting)
                enhanced_ndcg = adapter.calculate_enhanced_ndcg(relevance_scores, item_stock_counts, k=3)
                
                total_standard_ndcg += standard_ndcg
                total_enhanced_ndcg += enhanced_ndcg
                
                print(f"  Standard NDCG@3: {standard_ndcg:.4f}")
                print(f"  Enhanced NDCG@3: {enhanced_ndcg:.4f}")
                
                if standard_ndcg > 0:
                    improvement = ((enhanced_ndcg - standard_ndcg) / standard_ndcg) * 100
                    print(f"  Improvement: {improvement:.1f}%")
        
        # Calculate average scores
        avg_standard_ndcg = total_standard_ndcg / len(queries)
        avg_enhanced_ndcg = total_enhanced_ndcg / len(queries)
        
        print(f"\n--- Summary for Beta = {beta} ---")
        print(f"Average Standard NDCG@3: {avg_standard_ndcg:.4f}")
        print(f"Average Enhanced NDCG@3: {avg_enhanced_ndcg:.4f}")
        
        if avg_standard_ndcg > 0:
            overall_improvement = ((avg_enhanced_ndcg - avg_standard_ndcg) / avg_standard_ndcg) * 100
            print(f"Overall Improvement: {overall_improvement:.1f}%")
        
        print()
    
    # Demonstrate stock impact
    print("=== Stock Count Impact Demonstration ===")
    
    # Use beta = 0.3 for this demonstration
    adapter = CatalogInterpolationAdapter(catalog_embeddings, beta=0.3)
    
    # Mock scenario: high relevance items with different stock levels
    print("\nScenario: Two equally relevant items with different stock levels")
    relevance_scores = [1.0, 1.0, 0.5, 0.3, 0.1]
    stock_counts_high = [50, 2, 10, 5, 1]  # First item high stock, second low stock
    stock_counts_low = [2, 50, 10, 5, 1]   # Reversed stock levels
    
    ndcg_high_first = adapter.calculate_enhanced_ndcg(relevance_scores, stock_counts_high, k=5)
    ndcg_high_second = adapter.calculate_enhanced_ndcg(relevance_scores, stock_counts_low, k=5)
    
    print(f"NDCG with high stock item first: {ndcg_high_first:.4f}")
    print(f"NDCG with high stock item second: {ndcg_high_second:.4f}")
    
    improvement = ((ndcg_high_first - ndcg_high_second) / ndcg_high_second) * 100
    print(f"Stock-aware ranking improvement: {improvement:.1f}%")
    
    print("\nStock weight calculation:")
    for i, (rel, stock) in enumerate(zip(relevance_scores, stock_counts_high)):
        weight = np.log2(stock + 1)
        enhanced_rel = rel * weight
        print(f"  Item {i+1}: Relevance {rel:.1f} × log2({stock}+1) = {rel:.1f} × {weight:.2f} = {enhanced_rel:.2f}")
    
    print("\n=== Evaluation completed successfully! ===")


if __name__ == "__main__":
    run_adapter_evaluation()

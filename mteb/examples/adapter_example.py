"""
Example demonstrating the CatalogInterpolationAdapter with MockModel2.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path to import mteb modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mteb.models.adapter_models import CatalogInterpolationAdapter


class MockModel2:
    """Mock model that simulates embedding generation for testing."""
    
    def __init__(self, embedding_dim: int = 10):
        self.embedding_dim = embedding_dim
        np.random.seed(42)  # For reproducible results
    
    def encode(self, texts, **kwargs):
        """Generate mock embeddings for input texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Generate deterministic embeddings based on text hash
            hash_val = hash(text) % 1000000
            np.random.seed(hash_val)
            embedding = np.random.rand(self.embedding_dim)
            embeddings.append(embedding)
        
        return np.array(embeddings)


def run_adapter_demo():
    """Run the adapter demonstration."""
    print("=== CatalogInterpolationAdapter Demo with MockModel2 ===\n")
    
    # Initialize mock model
    model = MockModel2(embedding_dim=10)
    print("✓ MockModel2 initialized")
    
    # Create mock catalog data
    catalog_items = [
        "Summer Floral Dress - Beautiful floral dress perfect for summer occasions",
        "Leather Handbag - Premium leather handbag with multiple compartments", 
        "Casual Cotton T-Shirt - Comfortable cotton t-shirt for everyday wear",
        "Wool Knit Sweater - Cozy wool sweater for cold weather",
        "Designer Crossbody Bag - Stylish crossbody bag for modern women",
        "Knitted Cardigan - Soft knitted cardigan with button closure"
    ]
    
    # Generate catalog embeddings
    catalog_embeddings = model.encode(catalog_items)
    print(f"✓ Generated embeddings for {len(catalog_items)} catalog items")
    
    # Test queries
    test_queries = [
        "summer dress for party",
        "leather bag for work", 
        "casual shirt",
        "warm sweater",
        "stylish handbag"
    ]
    
    # Generate query embeddings
    query_embeddings = model.encode(test_queries)
    print(f"✓ Generated embeddings for {len(test_queries)} test queries\n")
    
    # Test different beta values
    beta_values = [0.0, 0.3, 0.7, 1.0]
    
    for beta in beta_values:
        print(f"--- Testing with Beta = {beta} ---")
        
        # Create adapter
        adapter = CatalogInterpolationAdapter(catalog_embeddings, beta=beta)
        
        # Test query enhancement
        for i, query in enumerate(test_queries):
            original_embedding = query_embeddings[i]
            enhanced_embedding = adapter.enhance_query_embedding(original_embedding)
            
            # Calculate similarity change
            original_norm = np.linalg.norm(original_embedding)
            enhanced_norm = np.linalg.norm(enhanced_embedding)
            similarity = np.dot(original_embedding, enhanced_embedding) / (original_norm * enhanced_norm)
            
            print(f"  Query: '{query}'")
            print(f"    Similarity to original: {similarity:.3f}")
            print(f"    Norm change: {original_norm:.3f} → {enhanced_norm:.3f}")
        
        print()
    
    # Test enhanced NDCG calculation
    print("--- Enhanced NDCG Calculation Test ---")
    
    # Mock relevance scores and product counts
    relevance_scores = [1.0, 0.8, 0.6, 0.4, 0.2]  # Decreasing relevance
    product_counts = [15, 8, 50, 12, 5]  # Varying stock levels
    
    adapter = CatalogInterpolationAdapter(catalog_embeddings, beta=0.3)
    
    # Calculate standard NDCG (without product count weighting)
    standard_relevance = relevance_scores.copy()
    standard_counts = [1] * len(relevance_scores)  # Equal weighting
    standard_ndcg = adapter.calculate_enhanced_ndcg(standard_relevance, standard_counts, k=5)
    
    # Calculate enhanced NDCG (with product count weighting)
    enhanced_ndcg = adapter.calculate_enhanced_ndcg(relevance_scores, product_counts, k=5)
    
    print(f"Standard NDCG@5: {standard_ndcg:.4f}")
    print(f"Enhanced NDCG@5: {enhanced_ndcg:.4f}")
    print(f"Improvement: {((enhanced_ndcg - standard_ndcg) / standard_ndcg * 100):.1f}%")
    
    print("\nProduct Count Impact:")
    for i, (rel, count) in enumerate(zip(relevance_scores, product_counts)):
        weight = np.log2(count + 1)
        enhanced_rel = rel * weight
        print(f"  Item {i+1}: Relevance {rel:.1f} × Stock Weight {weight:.2f} = {enhanced_rel:.2f}")
    
    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    run_adapter_demo()

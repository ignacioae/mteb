#!/usr/bin/env python3
"""
Example script demonstrating how to use the Google API encoder.

This script shows different ways to configure and use the GoogleAPIEncoder
for generating embeddings using Google Cloud Vertex AI APIs.

Requirements:
- Google Cloud project with Vertex AI API enabled
- Authentication set up (service account or gcloud auth)
- Required packages: google-cloud-aiplatform, vertexai

Usage:
    python google_api_example.py
"""

import os
import sys
import logging
from typing import List

# Add the parent directory to the path to import meb modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meb.models.encoders import get_encoder
from meb.models.google_api_encoder import GoogleAPIEncoder, create_google_api_encoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_simple_usage():
    """Example 1: Simple usage with environment variable."""
    print("\n=== Example 1: Simple Usage ===")
    
    # Set your Google Cloud project ID
    # You can also set this as an environment variable: export GOOGLE_CLOUD_PROJECT=your-project-id
    project_id = "your-project-id"  # Replace with your actual project ID
    
    if project_id == "your-project-id":
        print("Please set your Google Cloud project ID in the script")
        return
    
    try:
        # Create encoder using the factory function
        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
        encoder = get_encoder("google_api")
        
        # Test with sample data
        test_texts = [
            "A beautiful sunset over the mountains",
            "Modern smartphone with advanced camera features",
            "Delicious chocolate cake with strawberries"
        ]
        
        print(f"Encoding {len(test_texts)} texts...")
        text_embeddings = encoder.encode_text(test_texts)
        print(f"Text embeddings shape: {text_embeddings.shape}")
        
        # Print model information
        info = encoder.get_model_info()
        print(f"Using text model: {info['text_model']}")
        print(f"Text embedding dimension: {info['text_embedding_dimension']}")
        
    except Exception as e:
        print(f"Error in example 1: {e}")


def example_2_dictionary_configuration():
    """Example 2: Advanced configuration using dictionary."""
    print("\n=== Example 2: Dictionary Configuration ===")
    
    project_id = "your-project-id"  # Replace with your actual project ID
    
    if project_id == "your-project-id":
        print("Please set your Google Cloud project ID in the script")
        return
    
    try:
        # Advanced configuration
        encoder_config = {
            "type": "google_api",
            "project_id": project_id,
            "location": "us-central1",
            "text_model": "text-embedding-004",
            "image_model": "multimodalembedding@001",
            "rate_limit_requests_per_minute": 30,  # Conservative rate limit
            "batch_size": 3,
            "max_retries": 2,
            "name": "google_api_custom"
        }
        
        encoder = get_encoder(encoder_config)
        
        # Test with sample data
        test_texts = [
            "Red sports car on a highway",
            "Cat sleeping on a windowsill",
            "Fresh vegetables at a farmers market"
        ]
        
        print(f"Encoding {len(test_texts)} texts with custom configuration...")
        text_embeddings = encoder.encode_text(test_texts)
        print(f"Text embeddings shape: {text_embeddings.shape}")
        
        # Print configuration
        info = encoder.get_model_info()
        print(f"Rate limit: {info['rate_limit']} requests/minute")
        print(f"Batch size: {info['batch_size']}")
        
    except Exception as e:
        print(f"Error in example 2: {e}")


def example_3_direct_instantiation():
    """Example 3: Direct instantiation with custom parameters."""
    print("\n=== Example 3: Direct Instantiation ===")
    
    project_id = "your-project-id"  # Replace with your actual project ID
    
    if project_id == "your-project-id":
        print("Please set your Google Cloud project ID in the script")
        return
    
    try:
        # Direct instantiation
        encoder = GoogleAPIEncoder(
            project_id=project_id,
            location="us-central1",
            text_model="text-embedding-004",
            rate_limit_requests_per_minute=20,
            batch_size=2,
            name="direct_google_api"
        )
        
        # Test with sample data
        test_texts = [
            "Machine learning model for image classification",
            "Natural language processing with transformers"
        ]
        
        print(f"Encoding {len(test_texts)} texts with direct instantiation...")
        text_embeddings = encoder.encode_text(test_texts)
        print(f"Text embeddings shape: {text_embeddings.shape}")
        
        # Show embedding values (first few dimensions)
        print(f"First embedding (first 5 dims): {text_embeddings[0][:5]}")
        
    except Exception as e:
        print(f"Error in example 3: {e}")


def example_4_convenience_function():
    """Example 4: Using the convenience function."""
    print("\n=== Example 4: Convenience Function ===")
    
    project_id = "your-project-id"  # Replace with your actual project ID
    
    if project_id == "your-project-id":
        print("Please set your Google Cloud project ID in the script")
        return
    
    try:
        # Using convenience function
        encoder = create_google_api_encoder(
            project_id=project_id,
            rate_limit_requests_per_minute=15,
            batch_size=1
        )
        
        # Test with sample data
        test_texts = ["Single test sentence for embedding generation"]
        
        print(f"Encoding {len(test_texts)} text with convenience function...")
        text_embeddings = encoder.encode_text(test_texts)
        print(f"Text embeddings shape: {text_embeddings.shape}")
        
    except Exception as e:
        print(f"Error in example 4: {e}")


def example_5_image_encoding():
    """Example 5: Image encoding (requires actual image files)."""
    print("\n=== Example 5: Image Encoding ===")
    
    project_id = "your-project-id"  # Replace with your actual project ID
    
    if project_id == "your-project-id":
        print("Please set your Google Cloud project ID in the script")
        return
    
    try:
        encoder = create_google_api_encoder(
            project_id=project_id,
            rate_limit_requests_per_minute=10,
            batch_size=1
        )
        
        # Note: Replace these with actual image file paths
        test_images = [
            "sample_image1.jpg",  # Replace with actual image path
            "sample_image2.jpg"   # Replace with actual image path
        ]
        
        print(f"Attempting to encode {len(test_images)} images...")
        print("Note: This will fail if the image files don't exist")
        
        try:
            image_embeddings = encoder.encode_image(test_images)
            print(f"Image embeddings shape: {image_embeddings.shape}")
        except Exception as img_error:
            print(f"Image encoding failed (expected if images don't exist): {img_error}")
        
    except Exception as e:
        print(f"Error in example 5: {e}")


def example_6_caching_demo():
    """Example 6: Demonstrate caching functionality."""
    print("\n=== Example 6: Caching Demo ===")
    
    project_id = "your-project-id"  # Replace with your actual project ID
    
    if project_id == "your-project-id":
        print("Please set your Google Cloud project ID in the script")
        return
    
    try:
        encoder = create_google_api_encoder(
            project_id=project_id,
            name="caching_demo"
        )
        
        test_texts = ["Caching test sentence"]
        test_images = ["dummy_image.jpg"]  # Will create zero embeddings for missing images
        
        print("First call (will generate embeddings)...")
        text_emb1, img_emb1 = encoder.encode_dataset(
            test_texts, test_images, 
            dataset_name="test_dataset", 
            data_type="demo",
            use_cache=True
        )
        
        print("Second call (should load from cache)...")
        text_emb2, img_emb2 = encoder.encode_dataset(
            test_texts, test_images,
            dataset_name="test_dataset",
            data_type="demo", 
            use_cache=True
        )
        
        print(f"Embeddings are identical: {(text_emb1 == text_emb2).all()}")
        
    except Exception as e:
        print(f"Error in example 6: {e}")


def main():
    """Run all examples."""
    print("Google API Encoder Examples")
    print("=" * 50)
    
    print("\nBefore running these examples, make sure you have:")
    print("1. A Google Cloud project with Vertex AI API enabled")
    print("2. Authentication set up (gcloud auth application-default login)")
    print("3. Required packages installed (google-cloud-aiplatform, vertexai)")
    print("4. Updated the project_id in each example function")
    
    # Run examples
    example_1_simple_usage()
    example_2_dictionary_configuration()
    example_3_direct_instantiation()
    example_4_convenience_function()
    example_5_image_encoding()
    example_6_caching_demo()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()

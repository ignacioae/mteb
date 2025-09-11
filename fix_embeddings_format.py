#!/usr/bin/env python3
"""
Script to fix embedding format in provider dataset CSV files.
Converts multiline embeddings to single-line comma-separated format.
"""

import csv
import re
import os
import shutil
from typing import List

def clean_embedding_string(embedding_str: str) -> str:
    """
    Clean and format embedding string to proper CSV format.
    
    Args:
        embedding_str: Raw embedding string with potential newlines and spaces
        
    Returns:
        Cleaned embedding string in format "[val1, val2, val3, ...]"
    """
    if not embedding_str or embedding_str.strip() == '':
        return ''
    
    # Remove outer quotes if present
    embedding_str = embedding_str.strip().strip('"\'')
    
    # Extract content between brackets
    match = re.search(r'\[(.*?)\]', embedding_str, re.DOTALL)
    if not match:
        print(f"Warning: Could not find brackets in embedding: {embedding_str[:100]}...")
        return embedding_str
    
    content = match.group(1)
    
    # Split by whitespace and filter out empty strings
    values = [val.strip() for val in re.split(r'\s+', content) if val.strip()]
    
    # Join with commas and wrap in brackets
    return '[' + ', '.join(values) + ']'

def fix_csv_file(input_path: str, output_path: str = None) -> None:
    """
    Fix embedding format in a CSV file.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file (if None, overwrites input)
    """
    if output_path is None:
        # Create backup
        backup_path = input_path + '.backup'
        shutil.copy2(input_path, backup_path)
        print(f"Created backup: {backup_path}")
        output_path = input_path
    
    rows = []
    
    # Read the file
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for i, row in enumerate(reader):
            # Fix text_embedding if present
            if 'text_embedding' in row and row['text_embedding']:
                original = row['text_embedding']
                row['text_embedding'] = clean_embedding_string(original)
                if i == 0:  # Show example for first row
                    print(f"Text embedding example:")
                    print(f"  Original: {original[:100]}...")
                    print(f"  Fixed: {row['text_embedding'][:100]}...")
            
            # Fix image_embedding if present
            if 'image_embedding' in row and row['image_embedding']:
                original = row['image_embedding']
                row['image_embedding'] = clean_embedding_string(original)
                if i == 0:  # Show example for first row
                    print(f"Image embedding example:")
                    print(f"  Original: {original[:100]}...")
                    print(f"  Fixed: {row['image_embedding'][:100]}...")
            
            rows.append(row)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} rows...")
    
    # Write the fixed file
    print(f"Writing {output_path}...")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Fixed {len(rows)} rows in {output_path}")

def main():
    """Main function to fix provider dataset files."""
    
    # Fix catalog file
    catalog_path = 'mtbe/datasets/provider/catalog/provider_catalog.csv'
    if os.path.exists(catalog_path):
        print("Fixing catalog file...")
        fix_csv_file(catalog_path)
    else:
        print(f"Catalog file not found: {catalog_path}")
    
    # Fix test file
    test_path = 'mtbe/datasets/provider/test/provider_test.csv'
    if os.path.exists(test_path):
        print("\nFixing test file...")
        fix_csv_file(test_path)
    else:
        print(f"Test file not found: {test_path}")
    
    print("\nDone! Your embedding format has been fixed.")
    print("Backup files have been created with .backup extension.")

if __name__ == '__main__':
    main()

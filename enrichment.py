"""
Enrich extracted documents with image captions using Claude
"""
import os
from pathlib import Path

# Get current location
print(f"Current: {os.getcwd()}")

# Go up one level to project root
os.chdir('..')

# Verify
print(f"Changed to: {os.getcwd()}")
print(f"Files here: {os.listdir('.')}")

# Check if data folder exists now
print(f"data/processed exists: {Path('data/processed').exists()}")

# Now import will work
from image_captioner import ImageCaptioner
import json
from typing import List, Dict
from tqdm import tqdm




def enrich_documents(
    input_file: str = "data/processed/documents.json",
    output_file: str = "data/processed/enriched_documents.json"
) -> List[Dict]:
    """
    Load extracted documents and add captions to images
    
    Args:
        input_file: JSON file from document_processor
        output_file: Where to save enriched documents
    
    Returns:
        List of enriched documents
    """
    
    # Load extracted documents
    print(f"Loading documents from {input_file}...")
    with open(input_file, 'r') as f:
        documents = json.load(f)
    
    print(f"Loaded {len(documents)} documents")
    
    # Separate by type
    image_docs = [d for d in documents if d['type'] == 'image']
    text_docs = [d for d in documents if d['type'] == 'text']
    table_docs = [d for d in documents if d.get('type') == 'table']
    
    print(f"  - Text chunks: {len(text_docs)}")
    print(f"  - Images: {len(image_docs)}")
    print(f"  - Tables: {len(table_docs)}")
    
    if not image_docs:
        print("\n⚠️  No images to caption!")
        return documents
    
    # Initialize captioner
    print("\nInitializing Claude captioner...")
    captioner = ImageCaptioner()
    print("✅ Captioner ready")
    
    # Caption all images
    print(f"\n{'='*60}")
    print(f"Captioning {len(image_docs)} images with Claude Sonnet 4.5")
    print(f"{'='*60}")
    
    enriched_docs = []
    
    # Add text and table docs as-is
    for doc in text_docs + table_docs:
        enriched_docs.append(doc)
    
    # Caption images
    max_images_to_caption = 30
    for i, doc in enumerate(tqdm(image_docs[:max_images_to_caption], desc="Captioning images")):
        image_path = doc['content']
        source = doc['metadata'].get('source', 'unknown')
        page = doc['metadata'].get('page', 'unknown')
        
        # Check if image file exists
        if not Path(image_path).exists():
            print(f"\n⚠️  Image not found: {image_path}")
            continue
        
        # Generate caption
        context = f"Document: {source}, Page: {page}"
        
        try:
            caption = captioner.caption_image(image_path, context=context)
            
            # Store original image path in metadata
            doc['metadata']['image_path'] = doc['content']
            doc['metadata']['caption'] = caption
            
            # Replace content with caption (this will be embedded for search)
            doc['content'] = caption
            
            enriched_docs.append(doc)
            
            # Show progress every 10 images
            if (i + 1) % 10 == 0:
                print(f"\n✓ Captioned {i + 1}/{len(image_docs)} images")
                
        except Exception as e:
            print(f"\n❌ Error captioning {image_path}: {e}")
            # Add without caption
            enriched_docs.append(doc)
    
    # Save enriched documents
    print(f"\n{'='*60}")
    print(f"Saving enriched documents to {output_file}...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_docs, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enrichment complete!")
    print(f"{'='*60}")
    print(f"FINAL SUMMARY:")
    print(f"{'='*60}")
    print(f"Total documents: {len(enriched_docs)}")
    print(f"  - Text chunks: {len(text_docs)}")
    print(f"  - Images (with captions): {len([d for d in enriched_docs if d['type'] == 'image'])}")
    print(f"  - Tables: {len(table_docs)}")
    print(f"\nSaved to: {output_file}")
    
    return enriched_docs


if __name__ == "__main__":
    enriched_docs = enrich_documents()
import fitz
import os
from pathlib import Path
from typing import List, Dict
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from pdf2image import convert_from_path
from PIL import Image
import uuid
from tqdm import tqdm
import json

#This part processes pdf document by classifying it into different categories first 
class DocumentProcessor:
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text and embedded images using PyMuPDF
        More reliable than unstructured + poppler
        """
        documents = []
        filename = Path(pdf_path).stem
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text from page
                text = page.get_text()
                
                # Split text into reasonable chunks (by paragraphs)
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                for i, para in enumerate(paragraphs):
                    if len(para) > 10:  # Skip very short text
                        documents.append({
                            'type': 'text',
                            'content': para,
                            'metadata': {
                                'source': filename,
                                'page': page_num + 1,
                                'chunk_id': f"{filename}_p{page_num}_text_{i}",
                            }
                        })
                
                # Extract embedded images from page
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Save image
                        image_filename = f"{filename}_p{page_num}_img{img_index}.{image_ext}"
                        image_path = self.images_dir / image_filename
                        
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        documents.append({
                            'type': 'image',
                            'content': str(image_path),
                            'metadata': {
                                'source': filename,
                                'page': page_num + 1,
                                'chunk_id': f"{filename}_p{page_num}_img{img_index}",
                            }
                        })
                    except Exception as e:
                        print(f"  Warning: Could not extract image {img_index} from page {page_num}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            print(f"ERROR processing {filename}: {e}")
            return []
        
        return documents
    # This part ensures full recursive loop of pdf file loop - not a single file read 
    def process_directory(self, directory: str = "data/raw", 
                         recursive: bool = False) -> List[Dict]:
        """Process all PDFs in a directory"""
        all_documents = []
        
        if recursive:
            pdf_files = list(Path(directory).rglob("*.pdf"))
        else:
            pdf_files = list(Path(directory).glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {directory}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files\n")
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            docs = self.process_pdf(str(pdf_file))
            
            # Show progress for each file
            text_count = sum(1 for d in docs if d['type'] == 'text')
            img_count = sum(1 for d in docs if d['type'] == 'image')
            print(f"  ✓ {pdf_file.name}: {text_count} text chunks, {img_count} images")
            
            all_documents.extend(docs)
        
        return all_documents
    
    def save_to_json(self, documents: List[Dict], 
                     output_file: str = "data/processed/documents.json"):
        """Save extracted documents to JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(documents, f, indent=2)
        print(f"\n✅ Saved {len(documents)} documents to {output_file}")



#This part triggers the .py  --- process_directory to be going to cloud - do later

if __name__ == "__main__":
    processor = DocumentProcessor()
    documents = processor.process_directory("D:/Projects/mmRAG-with-Vision-Language-Model/PDFs")

    
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    #print("="*60)
    print(f"Total documents: {len(documents)}")
    print(f"Text chunks: {sum(1 for d in documents if d['type'] == 'text')}")
    print(f"Embedded images: {sum(1 for d in documents if d['type'] == 'image')}")
    print(f"\nImages location: {processor.images_dir}")
    
    if documents:
        processor.save_to_json(documents)
    else:
        print("\n⚠️  No documents extracted. Check your PDF files.")
        
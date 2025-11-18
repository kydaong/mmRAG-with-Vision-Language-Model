"""
RAG Query Engine - Combines retrieval with Claude for answers
Supports multimodal responses (returns text + relevant images)
"""
import anthropic
import base64
import os
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer



class MultimodalRAGEngine:
    def __init__(
        self,
        collection_name: str = "oil_gas_multimodal",
        qdrant_path: str = "./qdrant_data",
        use_server: bool = False,
        server_url: str = "http://localhost:6333"
    ):
        """
        Initialize RAG engine
        
        Args:
            collection_name: Qdrant collection name
            qdrant_path: Path to local Qdrant storage
            use_server: If True, connects to Qdrant server
            server_url: Qdrant server URL
        """
        load_dotenv()
        
        # Initialize Claude
        print("Initializing Claude...")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        self.claude = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        
        # Initialize Qdrant
        print("Connecting to Qdrant...")
        if use_server:
            self.qdrant = QdrantClient(url=server_url)
        else:
            self.qdrant = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        
        # Initialize embeddings
        print("Loading embeddings model...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print("✅ RAG Engine ready!")
    
    def search_documents(
        self, 
        query: str, 
        limit: int = 5,
        include_images: bool = True
    ) -> Dict:
        """
        Search for relevant documents in vector database
        
        Args:
            query: User question
            limit: Number of results to retrieve
            include_images: Whether to retrieve images or text only
        
        Returns:
            Dict with text_results and image_results
        """
        # Embed query
        query_vector = self.embedder.encode(query).tolist()
        
        # Search Qdrant
        results = self.qdrant.search(
            collection_name=self.collection_name, #this is where your vectors are at
            query_vector=query_vector,
            limit=limit * 2  # Get more results to separate text/images
        )
        
        # Separate text and images
        text_results = []
        image_results = []
        
        for result in results:
            doc_type = result.payload.get('type')
            
            if doc_type == 'text':
                text_results.append({
                    'score': result.score,
                    'content': result.payload['content'],
                    'source': result.payload['source'],
                    'page': result.payload.get('page')
                })
            elif doc_type == 'image' and include_images:
                image_results.append({
                    'score': result.score,
                    'caption': result.payload['content'],
                    'image_path': result.payload.get('image_path'),
                    'source': result.payload['source'],
                    'page': result.payload.get('page')
                })
        
        return {
            'text_results': text_results[:limit],
            'image_results': image_results[:3]  # Limit images (API cost)
        }
    
    def encode_image(self, image_path: str) -> tuple[str, str]:
        """Encode image to base64 for Claude"""
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        suffix = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return image_data, media_type_map.get(suffix, 'image/jpeg')
    
    def query(
        self,
        question: str,
        num_results: int = 5,
        include_images: bool = True,
        return_sources: bool = True
    ) -> Dict:
        """
        Main RAG query method
        
        Args:
            question: User question
            num_results: Number of documents to retrieve
            include_images: Whether to use images in answer
            return_sources: Whether to return source documents
        
        Returns:
            Dict with answer, sources, and retrieved images
        """
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print(f"{'='*60}")
        
        # Step 1: Retrieve relevant documents
        print("\n1. Searching vector database...")
        search_results = self.search_documents(
            query=question,
            limit=num_results,
            include_images=include_images
        )
        print(f"   ✓ Search complete") 
        
        text_docs = search_results['text_results']
        image_docs = search_results['image_results']
        
        print(f"   Found: {len(text_docs)} text chunks, {len(image_docs)} images")
        
        # Step 2: Build context for Claude
        print("\n2. Building context...")
        
        # Text context
        context_text = "\n\n".join([
            f"[Source: {doc['source']}, Page {doc['page']}]\n{doc['content']}"
            for doc in text_docs
        ])
        print(f"   ✓ Context built")
        
        # Build prompt
        prompt = f"""You are an expert Oil & Gas engineer assistant. You answer queries on plant equipment datasheet, plant processes and 
        various international standards. Answer the following question using the provided documentation."

**Question:** {question}

**Retrieved Documentation:**
{context_text}

**Instructions:**
- Provide a clear, technical answer
- Reference specific equipment, tag numbers, specifications when relevant
- Cite sources using [Source: document_name, Page X] format
- If images are provided, analyze them and incorporate visual information
- If you cannot fully answer from the provided context, say so

**Answer:**"""

        # Step 3: Prepare Claude API call with images
        print("\n3. Generating answer with Claude...")
        
        message_content = [{"type": "text", "text": prompt}]
        
        # Add images if available.
        images_added = []
        for img_doc in image_docs:
            image_path = img_doc['image_path']
            
            if image_path and Path(image_path).exists():
                try:
                    print(f"   Encoding: {Path(image_path).name}")
                    image_data, media_type = self.encode_image(image_path)
                    print(f"   ✓ Encoded ({len(image_data)} chars)")
                    
                    # Add image to message
                    message_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        }
                    })
                    
                    # Add caption context
                    message_content.append({
                        "type": "text",
                        "text": f"\n[Image from {img_doc['source']}, Page {img_doc['page']}]\nImage description: {img_doc['caption'][:200]}...\n"
                    })
                    
                    images_added.append(img_doc)
                    
                except Exception as e:
                    print(f"   Warning: Could not load image {image_path}: {e}")
        
        print(f"   Using {len(images_added)} images in context")
        
        # Call Claude
        
        response = self.claude.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.2,
            messages=[{"role": "user", "content": message_content}]
        )
        
        answer = response.content[0].text
        
    
        
        print(f"   Answer length: {len(answer)} chars")
        
        # Step 4: Format response
        result = {
            "question": question,
            "answer": answer,
        }
        
        if return_sources:
            result["sources"] = {
                "text_documents": [
                    {
                        "source": doc['source'],
                        "page": doc['page'],
                        "score": doc['score'],
                        "preview": doc['content'][:200] + "..."
                    }
                    for doc in text_docs
                ],
                "images": [
                    {
                        "source": img['source'],
                        "page": img['page'],
                        "score": img['score'],
                        "image_path": img['image_path']
                    }
                    for img in images_added
                ]
            }
        
        print(f"\n✅ Answer generated ({len(answer)} characters)")
        
        return result
    


if __name__ == "__main__":
    print("="*60)
    print("TESTING MULTIMODAL RAG ENGINE")
    print("="*60)
    
    # Initialize
    rag = MultimodalRAGEngine(
        qdrant_path="./qdrant_data",
        use_server=True  # Set to True if using Qdrant server
    )
    
    # Test queries
    test_queries = [
        "Talk about community risk management under element 10",
    ]
    
    for query in test_queries:
        result = rag.query(
            question=query,
            num_results=5,
            include_images=True
        )
        
        print(f"\n{'='*60}")
        print(f"Q: {result['question']}")
        print(f"{'='*60}")
        print(f"\nA: {result['answer']}")
        
        if 'sources' in result:
            print(f"\n📚 Sources:")
            print(f"   Text: {len(result['sources']['text_documents'])} documents")
            print(f"   Images: {len(result['sources']['images'])} images")
        
        print("\n" + "="*60)
        input("Press Enter for next query...")    




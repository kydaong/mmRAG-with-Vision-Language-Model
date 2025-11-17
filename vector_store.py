"""
Vector database setup with Qdrant
Stores both text and image captions for multimodal retrieval
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
#from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import uuid



class MultimodalVectorStore:
    def __init__(
        self,
        collection_name: str = "oil_gas_multimodal",
        use_local: bool = True,
        server_url: str = "http://localhost:6333",  #connection to qdrant server
    ):
        self.collection_name = collection_name
        """
        Initialize multimodal vector store with Qdrant
        Uses FREE sentence-transformers for embeddings
        """
        # Initialize Qdrant client
        if use_local:
            print("Initializing Qdrant (persistent disk storage)...")
            self.client = QdrantClient(path="./qdrant_data")
        else:
            print(f"Connecting to Qdrant server at {server_url}...")
            self.client = QdrantClient(url=server_url)

            try:
                collections = self.client.get_collections()
                print(f"✅ Connected! Server has {len(collections.collections)} collections")
            except Exception as e:
                print(f"❌ Connection failed: {e}")
                print("Make sure Qdrant server is running!")
                raise
        
        # Initialize FREE embeddings model - huggingface embedding model
        print("Loading sentence-transformers model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded - Embedding dimension: {self.embedding_dim}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts at once"""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed single query text"""
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()
        
    def create_collection(self):
        """Create Qdrant collection for multimodal documents"""
        print(f"\nCreating collection: {self.collection_name}")
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name in collection_names:
            print(f"Collection '{self.collection_name}' already exists. Deleting...")
            self.client.delete_collection(self.collection_name)
        
        # Create new collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Collection created: {self.collection_name}")
    
    def add_documents(self, documents: List[Dict], batch_size: int = 32):
        """
        Add documents to vector store
        
        Args:
            documents: List of enriched documents (text + image captions)
            batch_size: Number of documents to embed at once
        """
        print(f"\n{'='*60}")
        print(f"Adding {len(documents)} documents to vector store")
        print(f"{'='*60}")
        
        points = []
        
        # Process in batches
        for i in tqdm(range(0, len(documents), batch_size), desc="Embedding documents"):
            batch = documents[i:i + batch_size]
            
            # Extract content for embedding
            texts = [doc['content'] for doc in batch]
            
            # Generate embeddings for batch. It calls openAI embedding model and returns a list of vectors
            # Take note, at this point we are assuming we run finish "data_processed.py" and "enrichment.py" 
            # -- enrichment.py is the one that captions your extracted image and change its "contents" within the metadata
            # Need to check the json file whether the image captions has been captured under "contents" within the metadata
            embeddings = self.embed_documents(texts)
            
            # Create points
            # This part creates a glocally unique ID to prevent collisions in the vector database. its required by qdrant as a point identifier. 
            for doc, embedding in zip(batch, embeddings):
                point_id = str(uuid.uuid4())
                
                # Prepare payload
                payload = {
                    "content": doc['content'][:1000],  # Truncate for storage
                    "type": doc['type'],
                    "source": doc['metadata'].get('source', 'unknown'),
                    "page": doc['metadata'].get('page'),
                    "chunk_id": doc['metadata'].get('chunk_id', ''),
                }
                
                # Add image path if it's an image - add image specific metadata - so u know where is the original image
                if doc['type'] == 'image':
                    payload['image_path'] = doc['metadata'].get('image_path', '')
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
        
        # Upload all points to Qdrant
        print("\nUploading to Qdrant...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ Added {len(points)} documents to vector store")
        
        # Show statistics
        text_count = sum(1 for d in documents if d['type'] == 'text')
        image_count = sum(1 for d in documents if d['type'] == 'image')
        print(f"\nStatistics:")
        print(f"  - Text chunks: {text_count}")
        print(f"  - Images (with captions): {image_count}")
    
    def search(self, query: str, limit: int = 5, filter_type: Optional[str] = None) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            limit: Number of results to return
            filter_type: Optional filter by document type ('text' or 'image')
        
        Returns:
            List of relevant documents with scores
        """
        
        # Embed query
        query_embedding = self.embed_query(query)
        
        # Build filter
        query_filter = None
        if filter_type:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(
                must=[FieldCondition(key="type", match=MatchValue(value=filter_type))]
            )
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "score": result.score,
                "content": result.payload["content"],
                "type": result.payload["type"],
                "source": result.payload["source"],
                "page": result.payload.get("page"),
                "image_path": result.payload.get("image_path"),
            })
        
        return formatted_results
    

if __name__ == "__main__":
    print("="*60)
    print("BUILDING MULTIMODAL VECTOR DATABASE")
    print("Using sentence-transformers embeddings (huggingface)")
    print("="*60)
    
    # Load enriched documents
    enriched_file = "data/processed/enriched_documents.json"
    print(f"\nLoading enriched documents from {enriched_file}...")
    
    with open(enriched_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"Loaded {len(documents)} documents")
    
    # Initialize vector store
    vector_store = MultimodalVectorStore(
        collection_name="oil_gas_multimodal",
        use_local= False,
        server_url="http://localhost:6333"
    )
    
    # Create collection
    vector_store.create_collection()
    
    # Add documents
    vector_store.add_documents(documents)
    
    # Test search
    print("\n" + "="*60)
    print("TESTING SEARCH")
    print("="*60)
    
    test_queries = [
        "pump with bypass valve",
        "pressure relief valve",
        "safety equipment",
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-"*60)
        results = vector_store.search(query, limit=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result['type'].upper()}] Score: {result['score']:.3f}")
            print(f"   Source: {result['source']} (Page {result['page']})")
            print(f"   Content: {result['content'][:100]}...")
            if result['type'] == 'image':
                print(f"   Image: {result['image_path']}")
    
    print("\n" + "="*60)
    print("✅ Vector database built successfully!")
    print("="*60)

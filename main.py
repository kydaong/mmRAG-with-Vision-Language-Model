"""
FastAPI server for Multimodal RAG
Provides REST API for querying Oil & Gas documentation
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from rag_engineer import MultimodalRAGEngine

# Initialize FastAPI
app = FastAPI(
    title="Oil & Gas Multimodal RAG API",
    description="Query technical documentation using AI",
    version="1.0.0"
)

# Enable CORS for local machine access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine (singleton)
rag_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine
    print("Initializing RAG engine...")
    
    rag_engine = MultimodalRAGEngine(
        collection_name="oil_gas_multimodal",
        use_server=True,
        server_url="http://localhost:6333"
    )

    print("RAG engine ready!")

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    num_results: int = 5
    include_images: bool = True
    return_sources: bool = True

class Source(BaseModel):
    source: str
    page: Optional[int]
    score: float
    preview: Optional[str] = None

class ImageSource(BaseModel):
    source: str
    page: Optional[int]
    score: float
    image_path: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[Dict] = None
    processing_time: Optional[float] = None

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Multimodal RAG API",
        "version": "1.0.0"
    }

'''
@app.get("/")
async def root():
    """base health check endpoint"""
    return {
        "status": "healthy",
        "service": "oil and gas multimodal RAG API",
        "version": "1.0.0"
    }
'''

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return {
        "status": "healthy",
        "rag_engine": "initialized",
        "qdrant": "connected"
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system
    
    Args:
        request: QueryRequest with question and parameters
    
    Returns:
        QueryResponse with answer and sources
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        import time
        start_time = time.time()
        
        # Query RAG engine
        result = rag_engine.query(
            question=request.question,
            num_results=request.num_results,
            include_images=request.include_images,
            return_sources=request.return_sources
        )
        
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        # Get Qdrant collection info
        collection_info = rag_engine.qdrant.get_collection(rag_engine.collection_name)
        
        return {
            "collection_name": rag_engine.collection_name,
            "total_vectors": collection_info.points_count,
            "vector_dimension": collection_info.config.params.vectors.size,
            "status": "operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# For development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6999)
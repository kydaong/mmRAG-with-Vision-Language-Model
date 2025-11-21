#app.py combines main.py (fastapi) with the chainlit application (UI)
"""
Combined FastAPI + Chainlit Application
Oil & Gas Multimodal RAG System
"""
import os
import sys
from pathlib import Path

# Clear proxy variables
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        os.environ.pop(k, None)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import chainlit as cl
from chainlit.server import app as chainlit_app
import uvicorn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from rag_engineer import MultimodalRAGEngine

# Create FastAPI app
fastapi_app = FastAPI(
    title="Oil & Gas Multimodal RAG API",
    description="Query technical documentation using AI",
    version="1.0.0"
)

# Enable CORS
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG engine
rag_engine = None

# Request/Response Models - set the format
class QueryRequest(BaseModel):
    question: str
    num_results: int = 5
    include_images: bool = True
    return_sources: bool = True

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[Dict] = None
    processing_time: Optional[float] = None

# Creating Fastapi endpoints

@fastapi_app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine
    print("Initializing RAG engine...")
    
    rag_engine = MultimodalRAGEngine(
        collection_name="oil_gas_multimodal",
        use_server=True,
        server_url="http://localhost:6333")
    print("RAG engine ready!")

@fastapi_app.get("/")
async def root():
    """API health check"""
    return {
        "status": "API healthy",
        "service": "Oil & Gas Multimodal RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query" : "/query",
            "get_stats": "/stats"
        }
    }

@fastapi_app.get("/health")
async def health_check():
    """Detailed health check"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return {
        "status": "healthy",
        "rag_engine": "initialized",
        "qdrant": "connected"
    }

@fastapi_app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system"""
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

@fastapi_app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        collection_info = rag_engine.qdrant.get_collection(rag_engine.collection_name)
        
        return {
            "collection_name": rag_engine.collection_name,
            "total_vectors": collection_info.points_count,
            "vector_dimension": collection_info.config.params.vectors.size,
            "status": "operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# CHAINLIT UI - later mount chainlit on fastapi

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    if rag_engine is None:
        await cl.Message(
            content="RAG engine is not ready. Please refresh the page.",
            type="error"
        ).send()
        return
    
    await cl.Message(
        content="**Oil & Gas Technical Assistant Ready!**\n\n"
                "I can help you with:\n"
                "- Equipment specifications and P&IDs\n"
                "- Operating procedures and manuals\n"
                "- Safety protocols and standards\n"
                "- Technical documentation\n\n"
                "💡 **Try asking:**\n"
                "- *What is the design pressure for pump P-101?*\n"
                "- *Show me vertical turbine pump configuration*\n"
                "- *Explain Exxon OIMS framework*\n\n"
                "Ask me anything!"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages"""
    if rag_engine is None:
        await cl.Message(
            content="❌ System not ready. Please refresh the page.",
            type="error"
        ).send()
        return
    
    user_question = message.content
    
    # Show thinking indicator
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        import time
        start_time = time.time()
        
        # Query RAG engine directly
        result = rag_engine.query(
            question=user_question,
            num_results=5,
            include_images=True,
            return_sources=True
        )
        
        processing_time = time.time() - start_time
        
        # Format answer
        answer_text = result['answer']
        
        # Add sources
        if result.get('sources'):
            text_docs = result['sources'].get('text_documents', [])
            images = result['sources'].get('images', [])
            
            if text_docs or images:
                answer_text += f"\n\n---\n\n**📚 Sources:**"
                
                if text_docs:
                    answer_text += f"\n\n**Documents ({len(text_docs)}):**"
                    for i, doc in enumerate(text_docs[:3], 1):
                        answer_text += f"\n{i}. **{doc['source']}** (Page {doc['page']}) - Relevance: {doc['score']:.2%}"
                
                if images:
                    answer_text += f"\n\n**Images ({len(images)}):**"
                    for i, img in enumerate(images, 1):
                        answer_text += f"\n{i}. **{img['source']}** (Page {img['page']}) - Relevance: {img['score']:.2%}"
        
        # Add timing
        answer_text += f"\n\n*⏱️ Response time: {processing_time:.2f}s*"
        
        # Update message
        msg.content = answer_text
        await msg.update()
        
    except Exception as e:
        msg.content = f"**Error processing your question:**\n\n{str(e)}\n\nPlease try rephrasing your question."
        await msg.update()

# MOUNT CHAINLIT TO FASTAPI

fastapi_app.mount("/", chainlit_app)

# RUN SERVER

if __name__ == "__main__":
    print("="*60)
    print("Starting Oil & Gas RAG System")
    print("="*60)
    print("\nServices:")
    print("   - Fastapi:  http://localhost:6999")
    print("   - FastAPI docs: http://localhost:6999/")
    print("   - chainlit UI endpoint: http://localhost:6999/query")
    print("\n" + "="*60)
    
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=6999,
        log_level="info"
    )
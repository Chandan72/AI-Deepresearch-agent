
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import Optional, Dict, Any, Union
from config import Config

# Validate configuration before starting
"""if not Config.validate_config():
    print("\n🔧 Please configure your API keys and try again.")
    exit(1)"""

from research_agent import DeepResearchAgent
from model import QualityMetrics  # Import the model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deep Research Agent API",
    version="1.0.0",
    description="AI-powered research agent using LangGraph"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
research_agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the research agent on startup"""
    global research_agent
    try:
        logger.info("🚀 Initializing Deep Research Agent...")
        research_agent = DeepResearchAgent()
        logger.info("✅ Research Agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Research Agent: {e}")
        raise

class ResearchRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = {}

class ResearchResponse(BaseModel):
    success: bool
    report: Optional[str] = None
    citations: list = []
    quality_metrics: Optional[Dict[str, Any]] = None  # Keep as dict
    metadata: Dict[str, Any] = {}

# Helper function to convert QualityMetrics to dict
def convert_quality_metrics(quality_metrics: Any) -> Optional[Dict[str, Any]]:
    """Convert QualityMetrics object to dictionary"""
    if quality_metrics is None:
        return None
    
    if isinstance(quality_metrics, dict):
        return quality_metrics
    
    # Convert QualityMetrics object to dict
    if hasattr(quality_metrics, 'model_dump'):
        return quality_metrics.model_dump()
    elif hasattr(quality_metrics, 'dict'):
        return quality_metrics.dict()
    else:
        # Fallback: manually extract attributes
        try:
            return {
                "completeness": getattr(quality_metrics, 'completeness', 0.0),
                "accuracy": getattr(quality_metrics, 'accuracy', 0.0),
                "relevance": getattr(quality_metrics, 'relevance', 0.0),
                "coverage": getattr(quality_metrics, 'coverage', 0.0),
                "overall_score": getattr(quality_metrics, 'overall_score', 0.0)
            }
        except Exception:
            return {
                "completeness": 0.5,
                "accuracy": 0.5,
                "relevance": 0.5,
                "coverage": 0.5,
                "overall_score": 0.5
            }

# In-memory storage for async results
research_results = {}

@app.post("/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest):
    """Start a research task"""
    try:
        logger.info(f"🔍 Starting research for query: {request.query}")
        result = await research_agent.research(
            query=request.query,
            thread_id=request.thread_id
        )
        
        # **CRITICAL FIX** - Convert quality_metrics to dict
        quality_metrics_dict = convert_quality_metrics(result.get('quality_metrics'))
        
        # Create the response with proper data types
        response_data = {
            "success": result.get("success", False),
            "report": result.get("report"),
            "citations": result.get("citations", []),
            "quality_metrics": quality_metrics_dict,  # Now properly converted
            "metadata": result.get("metadata", {})
        }
        
        logger.info(f"✅ Research completed: {response_data['success']}")
        return ResearchResponse(**response_data)
        
    except Exception as e:
        logger.error(f"❌ Research failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/async")
async def start_async_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Start an asynchronous research task"""
    import uuid
    task_id = str(uuid.uuid4())
    
    # Store initial status
    research_results[task_id] = {
        "status": "started",
        "query": request.query,
        "result": None
    }
    
    # Start background research
    background_tasks.add_task(run_async_research, task_id, request)
    
    logger.info(f"🔄 Started async research task: {task_id}")
    return {"task_id": task_id, "status": "started"}

async def run_async_research(task_id: str, request: ResearchRequest):
    """Run research in background"""
    try:
        research_results[task_id]["status"] = "running"
        logger.info(f"🔍 Running async research: {task_id}")
        
        result = await research_agent.research(
            query=request.query,
            thread_id=request.thread_id
        )
        
        # Convert quality_metrics for async results too
        result["quality_metrics"] = convert_quality_metrics(result.get("quality_metrics"))
        
        research_results[task_id].update({
            "status": "completed",
            "result": result
        })
        
        logger.info(f"✅ Async research completed: {task_id}")
        
    except Exception as e:
        research_results[task_id].update({
            "status": "failed",
            "error": str(e)
        })
        logger.error(f"❌ Async research failed {task_id}: {e}")

@app.get("/research/status/{task_id}")
async def get_research_status(task_id: str):
    """Get status of an async research task"""
    if task_id not in research_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return research_results[task_id]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "config": {
            "model": Config.MODEL_NAME,
            "has_openai_key": bool(Config.OPENAI_API_KEY),
            "has_tavily_key": bool(Config.TAVILY_API_KEY)
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🔬 Deep Research Agent API",
        "version": "1.0.0",
        "status": "ready",
        "endpoints": {
            "research": "/research",
            "async_research": "/research/async",
            "status": "/research/status/{task_id}",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Deep Research Agent API...")
    print(f"📊 Configuration: Model={Config.MODEL_NAME}, Debug={Config.DEBUG}")
    
    uvicorn.run(
        "api:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
        log_level="info"
    )


from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import Annotated, TypedDict
from enum import Enum

class ResearchStatus(str, Enum):
    PLANNING = "planning"
    RESEARCHING = "researching"
    EVALUATING = "evaluating"
    GENERATING = "generating"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"

class ResearchStep(BaseModel):
    step_id: str = Field(description="Unique identifier for the research step")
    query: str = Field(description="Search query for this step")
    completed: bool = Field(default=False, description="Whether this step is completed")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    sources: List[str] = Field(default_factory=list, description="Source URLs")
    confidence: float = Field(default=0.0, description="Confidence score for this step")

class QualityMetrics(BaseModel):
    completeness: float = Field(ge=0, le=1, description="How complete is the research")
    accuracy: float = Field(ge=0, le=1, description="How accurate are the sources")
    relevance: float = Field(ge=0, le=1, description="How relevant to the query")
    coverage: float = Field(ge=0, le=1, description="How well does it cover the topic")
    overall_score: float = Field(ge=0, le=1, description="Overall quality score")

class AgentState(TypedDict):
    """State for the research agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Core research data
    query: str
    research_plan: List[ResearchStep]
    current_step: int
    
    # Research results
    all_sources: List[Dict[str, Any]]
    processed_content: List[Dict[str, Any]]
    
    # Quality and evaluation
    quality_metrics: Optional[QualityMetrics]
    needs_more_research: bool
    
    # Status and control
    status: ResearchStatus
    iteration_count: int
    max_iterations: int
    
    # Error handling
    errors: List[str]
    warnings: List[str]
    
    # Human feedback
    human_feedback: Optional[str]
    requires_human_review: bool
    
    # Final output
    final_report: Optional[str]
    citations: List[str]
    
    # Metadata
    start_time: Optional[str]
    end_time: Optional[str]
    total_tokens_used: int

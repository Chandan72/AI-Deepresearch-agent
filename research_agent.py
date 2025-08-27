
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
import logging
from models import AgentState, ResearchStatus, QualityMetrics
from tools import ResearchTools
from llm_handler import LLMHandler
from config import Config

logger = logging.getLogger(__name__)

class DeepResearchAgent:
    def __init__(self):
        self.tools = ResearchTools()
        self.llm_handler = LLMHandler()
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan_research", self._plan_research_node)
        workflow.add_node("conduct_research", self._conduct_research_node)
        workflow.add_node("evaluate_quality", self._evaluate_quality_node)
        workflow.add_node("generate_report", self._generate_report_node)
        workflow.add_node("human_review", self._human_review_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Define the workflow
        workflow.add_edge(START, "plan_research")
        workflow.add_edge("plan_research", "conduct_research")
        workflow.add_edge("conduct_research", "evaluate_quality")
        
        # Conditional routing after evaluation
        workflow.add_conditional_edges(
            "evaluate_quality",
            self._should_continue_research,
            {
                "continue": "conduct_research",
                "generate": "generate_report",
                "error": "error_handler"
            }
        )
        
        # Conditional routing after report generation
        workflow.add_conditional_edges(
            "generate_report",
            self._needs_human_review,
            {
                "review": "human_review",
                "complete": END,
                "error": "error_handler"
            }
        )
        
        workflow.add_edge("human_review", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _plan_research_node(self, state: AgentState) -> Dict[str, Any]:
        """Create a comprehensive research plan"""
        try:
            logger.info(f"Planning research for query: {state['query']}")
            
            research_plan = await self.llm_handler.create_research_plan(state["query"])
            
            return {
                "research_plan": research_plan,
                "current_step": 0,
                "status": ResearchStatus.RESEARCHING,
                "start_time": datetime.now().isoformat(),
                "iteration_count": 0,
                "max_iterations": Config.MAX_RESEARCH_ITERATIONS,
                "all_sources": [],
                "processed_content": [],
                "errors": [],
                "warnings": [],
                "needs_more_research": True,
                "requires_human_review": False,
                "total_tokens_used": self.llm_handler.token_usage
            }
            
        except Exception as e:
            logger.error(f"Research planning failed: {e}")
            return {
                "status": ResearchStatus.FAILED,
                "errors": [f"Planning failed: {str(e)}"]
            }
    
    async def _conduct_research_node(self, state: AgentState) -> Dict[str, Any]:
        """Conduct research for the current step"""
        try:
            current_step = state["current_step"]
            research_plan = state["research_plan"]
            
            if current_step >= len(research_plan):
                return {"needs_more_research": False}
            
            step = research_plan[current_step]
            logger.info(f"Researching step {current_step + 1}: {step.query}")
            
            async with self.tools as tools:
                # Perform web search
                search_results = await tools.web_search(step.query)
                
                # Scrape additional content from URLs
                urls_to_scrape = [result["url"] for result in search_results if result.get("url")]
                scraped_content = await tools.batch_scrape(urls_to_scrape[:5])  # Limit scraping
                
                # Update step with results
                step.results = search_results
                step.sources = urls_to_scrape
                step.completed = True
                step.confidence = min(len(search_results) / Config.MAX_SEARCH_RESULTS, 1.0)
                
                # Update research plan
                updated_plan = research_plan.copy()
                updated_plan[current_step] = step
                
                # Combine all collected data
                all_sources = state.get("all_sources", []) + search_results
                processed_content = state.get("processed_content", []) + scraped_content
                
                return {
                    "research_plan": updated_plan,
                    "current_step": current_step + 1,
                    "all_sources": all_sources,
                    "processed_content": processed_content,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "total_tokens_used": self.llm_handler.token_usage
                }
                
        except Exception as e:
            logger.error(f"Research conduct failed: {e}")
            errors = state.get("errors", [])
            errors.append(f"Research step {current_step} failed: {str(e)}")
            return {"errors": errors}
    
    async def _evaluate_quality_node(self, state: AgentState) -> Dict[str, Any]:
        """Evaluate research quality and determine next steps"""
        try:
            logger.info("Evaluating research quality")
            
            quality_metrics = await self.llm_handler.evaluate_research_quality(
                state["query"],
                state["all_sources"]
            )
            
            # Determine if more research is needed
            current_step = state["current_step"]
            plan_length = len(state["research_plan"])
            iteration_count = state.get("iteration_count", 0)
            max_iterations = state.get("max_iterations", Config.MAX_RESEARCH_ITERATIONS)
            
            needs_more = (
                current_step < plan_length or
                (quality_metrics.overall_score < Config.QUALITY_THRESHOLD and iteration_count < max_iterations)
            )
            
            return {
                "quality_metrics": quality_metrics,
                "needs_more_research": needs_more,
                "status": ResearchStatus.EVALUATING if needs_more else ResearchStatus.GENERATING,
                "total_tokens_used": self.llm_handler.token_usage
            }
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            errors = state.get("errors", [])
            errors.append(f"Quality evaluation failed: {str(e)}")
            return {"errors": errors}
    
    async def _generate_report_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate the final research report"""
        try:
            logger.info("Generating final report")
            
            quality_metrics = state.get("quality_metrics")
            if not quality_metrics:
                quality_metrics = QualityMetrics(
                    completeness=0.5, accuracy=0.5, relevance=0.5, coverage=0.5, overall_score=0.5
                )
            
            report, citations = await self.llm_handler.generate_final_report(
                state["query"],
                state["all_sources"],
                quality_metrics
            )
            
            # Determine if human review is needed
            requires_review = (
                quality_metrics.overall_score < 0.8 or
                len(state.get("errors", [])) > 0 or
                state.get("iteration_count", 0) >= Config.MAX_RESEARCH_ITERATIONS
            )
            
            return {
                "final_report": report,
                "citations": citations,
                "status": ResearchStatus.COMPLETED,
                "requires_human_review": requires_review,
                "end_time": datetime.now().isoformat(),
                "total_tokens_used": self.llm_handler.token_usage
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            errors = state.get("errors", [])
            errors.append(f"Report generation failed: {str(e)}")
            return {
                "errors": errors,
                "status": ResearchStatus.FAILED
            }
    
    async def _human_review_node(self, state: AgentState) -> Dict[str, Any]:
        """Handle human review process"""
        # This is a placeholder for human-in-the-loop functionality
        # In production, this would integrate with your UI/API for human feedback
        
        logger.info("Human review step - awaiting feedback")
        
        return {
            "status": ResearchStatus.REVIEWING,
            "human_feedback": "Pending human review"
        }
    
    async def _error_handler_node(self, state: AgentState) -> Dict[str, Any]:
        """Handle errors and provide graceful degradation"""
        errors = state.get("errors", [])
        logger.error(f"Research agent encountered errors: {errors}")
        
        # Generate a minimal error report
        error_report = f"""# Research Report - Error Occurred

## Query
{state.get('query', 'Unknown query')}

## Error Summary
The research process encountered the following errors:
{chr(10).join(f"- {error}" for error in errors)}

## Partial Results
{len(state.get('all_sources', []))} sources were collected before the error occurred.

## Recommendations
1. Try simplifying the research query
2. Check your API keys and network connection
3. Contact support if errors persist
"""
        
        return {
            "final_report": error_report,
            "status": ResearchStatus.FAILED,
            "end_time": datetime.now().isoformat()
        }
    
    def _should_continue_research(self, state: AgentState) -> str:
        """Determine whether to continue research or move to report generation"""
        if state.get("errors"):
            return "error"
        
        if state.get("needs_more_research", True):
            return "continue"
        else:
            return "generate"
    
    def _needs_human_review(self, state: AgentState) -> str:
        """Determine if human review is needed"""
        if state.get("errors"):
            return "error"
        
        if state.get("requires_human_review", False):
            return "review"
        else:
            return "complete"
    
    async def research(self, query: str, thread_id: str = None) -> Dict[str, Any]:
        """Main research method"""
        if not thread_id:
            thread_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "query": query,
            "messages": []
        }
        
        try:
            # Run the research workflow
            final_state = await self.graph.ainvoke(initial_state, config)
            
            return {
                "success": final_state.get("status") == ResearchStatus.COMPLETED,
                "report": final_state.get("final_report"),
                "citations": final_state.get("citations", []),
                "quality_metrics": final_state.get("quality_metrics"),
                "metadata": {
                    "thread_id": thread_id,
                    "start_time": final_state.get("start_time"),
                    "end_time": final_state.get("end_time"),
                    "total_tokens": final_state.get("total_tokens_used", 0),
                    "sources_count": len(final_state.get("all_sources", [])),
                    "errors": final_state.get("errors", []),
                    "warnings": final_state.get("warnings", [])
                }
            }
            
        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            return {
                "success": False,
                "report": f"Research failed: {str(e)}",
                "citations": [],
                "quality_metrics": None,
                "metadata": {
                    "thread_id": thread_id,
                    "error": str(e)
                }
            }

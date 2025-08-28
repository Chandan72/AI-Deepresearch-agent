
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import logging
from model import AgentState, ResearchStatus, QualityMetrics
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
        """Build the LangGraph workflow with proper recursion control"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan_research", self._plan_research_node)
        workflow.add_node("conduct_research", self._conduct_research_node)
        workflow.add_node("evaluate_quality", self._evaluate_quality_node)
        workflow.add_node("generate_report", self._generate_report_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Define the workflow with better control
        workflow.add_edge(START, "plan_research")
        workflow.add_edge("plan_research", "conduct_research")
        workflow.add_edge("conduct_research", "evaluate_quality")
        
        # **CRITICAL FIX** - Better conditional routing
        workflow.add_conditional_edges(
            "evaluate_quality",
            self._should_continue_research,
            {
                "continue": "conduct_research",
                "generate": "generate_report", 
                "complete": "generate_report",  # Add this path
                "error": "error_handler"
            }
        )
        
        workflow.add_edge("generate_report", END)
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
            iteration_count = state.get("iteration_count", 0)
            
            logger.info(f"Researching step {current_step + 1}/{len(research_plan)} (iteration {iteration_count})")
            
            # **CRITICAL FIX** - Check limits early
            if current_step >= len(research_plan):
                logger.info("All research steps completed")
                return {
                    "needs_more_research": False,
                    "current_step": current_step,
                    "iteration_count": iteration_count + 1
                }
            
            if iteration_count >= Config.MAX_RESEARCH_ITERATIONS:
                logger.warning(f"Max iterations ({Config.MAX_RESEARCH_ITERATIONS}) reached")
                return {
                    "needs_more_research": False,
                    "warnings": state.get("warnings", []) + ["Maximum research iterations reached"],
                    "iteration_count": iteration_count + 1
                }
            
            step = research_plan[current_step]
            
            async with self.tools as tools:
                # Perform web search
                search_results = await tools.web_search(step.query)
                
                # Scrape additional content from URLs
                urls_to_scrape = [result["url"] for result in search_results if result.get("url")][:3]  # Limit to 3
                scraped_content = await tools.batch_scrape(urls_to_scrape)
                
                # Update step with results
                step.results = search_results
                step.sources = urls_to_scrape
                step.completed = True
                step.confidence = min(len(search_results) / max(Config.MAX_SEARCH_RESULTS, 1), 1.0)
                
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
                    "iteration_count": iteration_count + 1,
                    "total_tokens_used": self.llm_handler.token_usage
                }
                
        except Exception as e:
            logger.error(f"Research conduct failed: {e}")
            errors = state.get("errors", [])
            errors.append(f"Research step {current_step} failed: {str(e)}")
            return {
                "errors": errors,
                "iteration_count": state.get("iteration_count", 0) + 1
            }
    
    async def _evaluate_quality_node(self, state: AgentState) -> Dict[str, Any]:
        """Evaluate research quality and determine next steps"""
        try:
            iteration_count = state.get("iteration_count", 0)
            logger.info(f"Evaluating research quality (iteration {iteration_count})")
            
            # **CRITICAL FIX** - Force completion after max iterations
            if iteration_count >= Config.MAX_RESEARCH_ITERATIONS:
                logger.warning("Forcing completion due to max iterations")
                return {
                    "quality_metrics": QualityMetrics(
                        completeness=0.7, accuracy=0.7, relevance=0.7, 
                        coverage=0.7, overall_score=0.7
                    ),
                    "needs_more_research": False,
                    "status": ResearchStatus.GENERATING,
                    "warnings": state.get("warnings", []) + ["Research completed due to iteration limit"],
                    "total_tokens_used": self.llm_handler.token_usage
                }
            
            # Skip LLM evaluation if we have enough sources
            all_sources = state.get("all_sources", [])
            if len(all_sources) >= 5:  # Good enough threshold
                logger.info(f"Sufficient sources found ({len(all_sources)}), proceeding to report")
                return {
                    "quality_metrics": QualityMetrics(
                        completeness=0.8, accuracy=0.8, relevance=0.8,
                        coverage=0.8, overall_score=0.8
                    ),
                    "needs_more_research": False,
                    "status": ResearchStatus.GENERATING,
                    "total_tokens_used": self.llm_handler.token_usage
                }
            
            # **SIMPLIFIED EVALUATION** - Avoid complex LLM calls
            current_step = state["current_step"]
            plan_length = len(state["research_plan"])
            
            # Simple logic: continue if more steps remain AND under iteration limit
            needs_more = (
                current_step < plan_length and 
                iteration_count < Config.MAX_RESEARCH_ITERATIONS and
                len(all_sources) < 3  # Only continue if very few sources
            )
            
            quality_score = min(len(all_sources) / 5.0, 1.0)  # Simple quality based on source count
            
            return {
                "quality_metrics": QualityMetrics(
                    completeness=quality_score,
                    accuracy=0.7,
                    relevance=0.7,
                    coverage=quality_score,
                    overall_score=quality_score
                ),
                "needs_more_research": needs_more,
                "status": ResearchStatus.EVALUATING if needs_more else ResearchStatus.GENERATING,
                "total_tokens_used": self.llm_handler.token_usage
            }
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return {
                "quality_metrics": QualityMetrics(
                    completeness=0.5, accuracy=0.5, relevance=0.5,
                    coverage=0.5, overall_score=0.5
                ),
                "needs_more_research": False,  # Stop on error
                "status": ResearchStatus.GENERATING,
                "errors": state.get("errors", []) + [f"Quality evaluation failed: {str(e)}"],
                "total_tokens_used": self.llm_handler.token_usage
            }
    
    async def _generate_report_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate the final research report"""
        try:
            logger.info("Generating final report")
            
            quality_metrics = state.get("quality_metrics")
            if not quality_metrics:
                quality_metrics = QualityMetrics(
                    completeness=0.5, accuracy=0.5, relevance=0.5, coverage=0.5, overall_score=0.5
                )
            
            all_sources = state.get("all_sources", [])
            if not all_sources:
                # Create a basic report even without sources
                report = f"""# Research Report: {state['query']}

## Summary
Research was conducted on the topic "{state['query']}" but limited sources were available.

## Findings
Due to technical limitations or API restrictions, comprehensive research results were not obtained. This may be due to:
- API key limitations
- Network connectivity issues
- Search API restrictions

## Recommendations
1. Verify API keys are properly configured
2. Check network connectivity
3. Try a more specific research query
4. Contact support if issues persist

## Metadata
- Sources collected: {len(all_sources)}
- Research iterations: {state.get('iteration_count', 0)}
- Status: Completed with limitations
"""
                citations = []
            else:
                report, citations = await self.llm_handler.generate_final_report(
                    state["query"],
                    all_sources,
                    quality_metrics
                )
            
            return {
                "final_report": report,
                "citations": citations,
                "status": ResearchStatus.COMPLETED,
                "requires_human_review": False,  # Simplified
                "end_time": datetime.now().isoformat(),
                "total_tokens_used": self.llm_handler.token_usage
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            # Create error report
            error_report = f"""# Research Report Error

## Query
{state.get('query', 'Unknown query')}

## Error Summary
Report generation failed: {str(e)}

## Available Data
- Sources collected: {len(state.get('all_sources', []))}
- Research steps completed: {state.get('current_step', 0)}

## Recommendations
Please try again with a simpler query or contact support.
"""
            
            return {
                "final_report": error_report,
                "citations": [],
                "status": ResearchStatus.FAILED,
                "errors": state.get("errors", []) + [f"Report generation failed: {str(e)}"],
                "end_time": datetime.now().isoformat(),
                "total_tokens_used": self.llm_handler.token_usage
            }
    
    async def _error_handler_node(self, state: AgentState) -> Dict[str, Any]:
        """Handle errors and provide graceful degradation"""
        errors = state.get("errors", [])
        logger.error(f"Research agent encountered errors: {errors}")
        
        error_report = f"""# Research Report - Error Occurred

## Query
{state.get('query', 'Unknown query')}

## Error Summary
The research process encountered errors:
{chr(10).join(f"- {error}" for error in errors)}

## Partial Results
- Sources collected: {len(state.get('all_sources', []))}
- Research iterations: {state.get('iteration_count', 0)}

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
        """**CRITICAL FIX** - Improved decision logic"""
        
        # Check for errors first
        if state.get("errors"):
            logger.info("Routing to error handler due to errors")
            return "error"
        
        # Check iteration limit
        iteration_count = state.get("iteration_count", 0)
        if iteration_count >= Config.MAX_RESEARCH_ITERATIONS:
            logger.info(f"Max iterations ({Config.MAX_RESEARCH_ITERATIONS}) reached, generating report")
            return "generate"
        
        # Check if research is needed
        needs_more = state.get("needs_more_research", False)
        current_step = state.get("current_step", 0)
        plan_length = len(state.get("research_plan", []))
        
        # Force completion if all steps done
        if current_step >= plan_length:
            logger.info("All research steps completed, generating report")
            return "generate"
        
        # Continue only if explicitly needed and under limits
        if needs_more and iteration_count < Config.MAX_RESEARCH_ITERATIONS:
            logger.info(f"Continuing research: step {current_step}/{plan_length}, iteration {iteration_count}")
            return "continue"
        else:
            logger.info("Research complete, generating report")
            return "generate"
        
    # research_agent.py - Fix the research method
    # research_agent.py - Fix the research method
    async def research(self, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Main research method with proper error handling"""
        if not thread_id:
            thread_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configuration with recursion limit
        config: Dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": Config.MAX_RESEARCH_ITERATIONS + 5
        }
        
        # Initial state
        initial_state: Dict[str, Any] = {
            "query": query,
            "messages": []
        }
        
        try:
            logger.info(f"Starting research with recursion limit: {config['recursion_limit']}")
            
            # **CRITICAL FIX** - Add proper error handling around graph invocation
            final_state = await self.graph.ainvoke(initial_state, config)
            
            # **CRITICAL FIX** - Check if final_state is None
            if final_state is None:
                logger.error("❌ LangGraph returned None state")
                return {
                    "success": False,
                    "report": f"""# Research Failed - System Error

    ## Query
    {query}

    ## Error
    The research workflow completed but returned an invalid state.

    ## Troubleshooting
    1. Check LangGraph configuration
    2. Verify all node implementations
    3. Review workflow definitions
    4. Contact support if issue persists
    """,
                    "citations": [],
                    "quality_metrics": None,
                    "metadata": {
                        "thread_id": thread_id,
                        "error": "LangGraph returned None state"
                    }
                }
            
            # **CRITICAL FIX** - Add type checking and safe access
            logger.info(f"✅ Workflow completed, processing final state...")
            
            # Safe access to final_state attributes
            try:
                status = final_state.get("status") if isinstance(final_state, dict) else None
                success = status in [ResearchStatus.COMPLETED, ResearchStatus.REVIEWING] if status else False
                
                # Safe access to quality_metrics
                quality_metrics = final_state.get("quality_metrics") if isinstance(final_state, dict) else None
                quality_metrics_dict = None
                
                if quality_metrics:
                    try:
                        if hasattr(quality_metrics, 'model_dump'):
                            quality_metrics_dict = quality_metrics.model_dump()
                        elif hasattr(quality_metrics, 'dict'):
                            quality_metrics_dict = quality_metrics.dict()
                        else:
                            quality_metrics_dict = {
                                "completeness": getattr(quality_metrics, 'completeness', 0.0),
                                "accuracy": getattr(quality_metrics, 'accuracy', 0.0),
                                "relevance": getattr(quality_metrics, 'relevance', 0.0),
                                "coverage": getattr(quality_metrics, 'coverage', 0.0),
                                "overall_score": getattr(quality_metrics, 'overall_score', 0.0)
                            }
                    except Exception as qm_error:
                        logger.warning(f"⚠️ Quality metrics conversion failed: {qm_error}")
                        quality_metrics_dict = {
                            "completeness": 0.7, "accuracy": 0.7, "relevance": 0.7,
                            "coverage": 0.7, "overall_score": 0.7
                        }
                
                # Safe access to other fields
                report = final_state.get("final_report", "No report generated") if isinstance(final_state, dict) else "State access error"
                citations = final_state.get("citations", []) if isinstance(final_state, dict) else []
                all_sources = final_state.get("all_sources", []) if isinstance(final_state, dict) else []
                
                return {
                    "success": success,
                    "report": report,
                    "citations": citations,
                    "quality_metrics": quality_metrics_dict,
                    "metadata": {
                        "thread_id": thread_id,
                        "start_time": final_state.get("start_time") if isinstance(final_state, dict) else None,
                        "end_time": final_state.get("end_time") if isinstance(final_state, dict) else None,
                        "total_tokens": final_state.get("total_tokens_used", 0) if isinstance(final_state, dict) else 0,
                        "sources_count": len(all_sources),
                        "iterations": final_state.get("iteration_count", 0) if isinstance(final_state, dict) else 0,
                        "errors": final_state.get("errors", []) if isinstance(final_state, dict) else [],
                        "warnings": final_state.get("warnings", []) if isinstance(final_state, dict) else []
                    }
                }
                
            except Exception as state_error:
                logger.error(f"❌ Error processing final state: {state_error}")
                return {
                    "success": False,
                    "report": f"""# Research State Processing Error

    ## Query
    {query}

    ## Issue
    The research completed successfully but there was an error processing the final results.

    ## Error Details
    {str(state_error)}

    ## Status
    Research data was collected but final formatting failed.
    """,
                    "citations": [],
                    "quality_metrics": None,
                    "metadata": {
                        "thread_id": thread_id,
                        "error": f"State processing error: {str(state_error)}"
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Research workflow failed: {e}")
            logger.error(f"❌ Exception type: {type(e)}")
            logger.error(f"❌ Exception args: {e.args}")
            
            return {
                "success": False,
                "report": f"""# Research Workflow Failed

    ## Query
    {query}

    ## Error  
    {str(e)}

    ## Error Type
    {type(e).__name__}

    ## Troubleshooting
    1. Check your API keys are valid
    2. Verify network connectivity
    3. Try a simpler query
    4. Review the logs for more details
    5. Contact support if issues persist

    ## Debug Info
    - Thread ID: {thread_id}
    - Error occurred during workflow execution
    """,
                "citations": [],
                "quality_metrics": None,
                "metadata": {
                    "thread_id": thread_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            }

            
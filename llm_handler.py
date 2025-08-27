# 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Optional
import json
from config import Config
from models import ResearchStep, QualityMetrics

class LLMHandler:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            api_key=Config.OPENAI_API_KEY
        )
        self.token_usage = 0
    
    async def create_research_plan(self, query: str) -> List[ResearchStep]:
        """Generate a comprehensive research plan for the given query"""
        
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research planning expert. Given a user query, create a comprehensive research plan.
            
Your task is to break down the query into 3-7 specific research steps that will provide complete coverage of the topic.
Each step should have a focused search query that will yield relevant information.

Consider:
- What are the key aspects of this topic?
- What background information is needed?
- What current developments should be researched?
- What different perspectives exist?
- What quantitative data might be relevant?

Return your response as a JSON array of research steps, each with:
- step_id: unique identifier (step_1, step_2, etc.)
- query: specific search query for this step
- completed: false (always start as false)
- results: [] (empty array initially)
- sources: [] (empty array initially)
- confidence: 0.0 (start at 0)

Example format:
[
  {{"step_id": "step_1", "query": "background information about X", "completed": false, "results": [], "sources": [], "confidence": 0.0}},
  {{"step_id": "step_2", "query": "current trends in X industry", "completed": false, "results": [], "sources": [], "confidence": 0.0}}
]"""),
            ("human", "Create a research plan for: {query}")
        ])
        
        try:
            response = await self.llm.ainvoke(planning_prompt.format_messages(query=query))
            self.token_usage += response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            
            # Parse the JSON response
            plan_data = json.loads(response.content)
            return [ResearchStep(**step) for step in plan_data]
            
        except Exception as e:
            # Fallback to a simple plan if LLM fails
            return [
                ResearchStep(
                    step_id="step_1",
                    query=f"comprehensive overview of {query}",
                    completed=False,
                    results=[],
                    sources=[],
                    confidence=0.0
                )
            ]
    
    async def evaluate_research_quality(
        self, 
        query: str, 
        research_data: List[Dict[str, Any]]
    ) -> QualityMetrics:
        """Evaluate the quality and completeness of research data"""
        
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research quality evaluator. Analyze the provided research data and evaluate its quality across multiple dimensions.

Evaluate the research on these criteria (0.0 to 1.0 scale):
1. Completeness: How thoroughly does the research cover the topic?
2. Accuracy: How reliable and trustworthy are the sources?
3. Relevance: How relevant is the information to the original query?
4. Coverage: How well does it address different aspects of the topic?

Consider:
- Source diversity and credibility
- Information recency and relevance
- Coverage of different perspectives
- Depth of information
- Gaps that might exist

Return your evaluation as JSON:
{{
  "completeness": 0.0-1.0,
  "accuracy": 0.0-1.0,
  "relevance": 0.0-1.0,
  "coverage": 0.0-1.0,
  "overall_score": 0.0-1.0
}}"""),
            ("human", """Original query: {query}

Research data summary:
{research_summary}

Evaluate this research quality.""")
        ])
        
        try:
            # Create a summary of research data
            research_summary = self._create_research_summary(research_data)
            
            response = await self.llm.ainvoke(
                evaluation_prompt.format_messages(
                    query=query,
                    research_summary=research_summary
                )
            )
            self.token_usage += response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            
            metrics_data = json.loads(response.content)
            return QualityMetrics(**metrics_data)
            
        except Exception as e:
            # Fallback to neutral metrics
            return QualityMetrics(
                completeness=0.5,
                accuracy=0.5,
                relevance=0.5,
                coverage=0.5,
                overall_score=0.5
            )
    
    async def generate_final_report(
        self, 
        query: str, 
        research_data: List[Dict[str, Any]],
        quality_metrics: QualityMetrics
    ) -> tuple[str, List[str]]:
        """Generate a comprehensive final report with citations"""
        
        report_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research analyst and report writer. Create a comprehensive, well-structured report based on the provided research data.

Report Structure:
1. Executive Summary
2. Background and Context
3. Key Findings (organize by themes/topics)
4. Analysis and Insights
5. Implications and Conclusions
6. Recommendations (if applicable)

Writing Guidelines:
- Use clear, professional language
- Organize information logically
- Include specific data points and examples
- Maintain objectivity and balance
- Cite sources using , ,  format
- Ensure the report directly addresses the original query

Quality Metrics Context:
Completeness: {completeness}
Accuracy: {accuracy}
Relevance: {relevance}
Coverage: {coverage}
Overall Score: {overall_score}

If quality scores are low, acknowledge limitations in your report."""),
            ("human", """Original Query: {query}

Research Data:
{research_data}

Generate a comprehensive research report with proper citations.""")
        ])
        
        try:
            research_content = self._format_research_for_report(research_data)
            
            response = await self.llm.ainvoke(
                report_prompt.format_messages(
                    query=query,
                    research_data=research_content,
                    completeness=quality_metrics.completeness,
                    accuracy=quality_metrics.accuracy,
                    relevance=quality_metrics.relevance,
                    coverage=quality_metrics.coverage,
                    overall_score=quality_metrics.overall_score
                )
            )
            self.token_usage += response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            
            report = response.content
            
            # Extract citations from the research data
            citations = []
            for idx, item in enumerate(research_data, 1):
                if item.get('url'):
                    citations.append(f"[{idx}] {item.get('title', 'Untitled')} - {item['url']}")
            
            return report, citations
            
        except Exception as e:
            error_report = f"""# Research Report Error

An error occurred while generating the final report for the query: "{query}"

Error: {str(e)}

## Available Data Summary
{len(research_data)} sources were collected but could not be properly formatted into a report.

Please try again or contact support if this issue persists."""
            
            return error_report, []
    
    def _create_research_summary(self, research_data: List[Dict[str, Any]]) -> str:
        """Create a concise summary of research data for evaluation"""
        summary_parts = []
        for item in research_data:
            if item.get('content'):
                summary_parts.append(f"Source: {item.get('title', 'Unknown')}\nContent snippet: {item['content'][:200]}...\n")
        return "\n".join(summary_parts[:10])  # Limit to first 10 sources
    
    def _format_research_for_report(self, research_data: List[Dict[str, Any]]) -> str:
        """Format research data for report generation"""
        formatted_data = []
        for idx, item in enumerate(research_data, 1):
            formatted_item = f"""[{idx}] {item.get('title', 'Untitled')}
URL: {item.get('url', 'N/A')}
Content: {item.get('content', 'No content available')[:1000]}...
"""
            formatted_data.append(formatted_item)
        return "\n".join(formatted_data)

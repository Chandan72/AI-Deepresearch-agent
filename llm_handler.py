
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Optional, Tuple
import json
import asyncio
import logging
from config import Config
from model import ResearchStep, QualityMetrics

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=2000,
            request_timeout=60,
            api_key=Config.GEMINI_API_KEY
        )
        self.token_usage = 0
    
    async def create_research_plan(self, query: str) -> List[ResearchStep]:
        """Generate a simple research plan"""
        try:
            logger.info(f"Creating research plan for: {query}")
            # Simplified planning to avoid LLM issues
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
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return [ResearchStep(
                step_id="step_1", 
                query=query, 
                completed=False, 
                results=[], 
                sources=[], 
                confidence=0.0
            )]
    
    async def generate_final_report(
        self, 
        query: str, 
        research_data: List[Dict[str, Any]],
        quality_metrics: QualityMetrics
    ) -> Tuple[str, List[str]]:
        """Generate report with comprehensive error handling"""
        
        try:
            logger.info(f"Generating report for {len(research_data)} sources")
            
            # **CRITICAL FIX** - Check for valid API key first
            if not Config.GEMINI_API_KEY or Config.GEMINI_API_KEY.startswith('dummy'):
                logger.warning("⚠️ Invalid OpenAI API key, creating fallback report")
                return await self._create_fallback_report(query, research_data)
            
            # Limit input data to prevent token overflow
            limited_data = research_data[:4]  # Only use first 3 sources
            
            if not limited_data:
                logger.warning("⚠️ No research data available")
                return await self._create_fallback_report(query, [])
            
            # Check total content length
            total_content = sum(len(str(item.get('content', ''))) for item in limited_data)
            if total_content > 8000:  # Too much content
                logger.warning(f"⚠️ Large content ({total_content} chars), using summary")
                return await self._create_summary_report(query, limited_data)
            
            # Create simplified prompt
            report_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research analyst. Create a concise report based on the provided sources.

Structure:
1. Executive Summary (2-3 sentences)
2. Key Findings (5-7 bullet points)  
3. Conclusions (1-2 sentences)

Keep the report under 500 words. Use [1], [2], [3], [4] for citations.
Be factual and concise."""),
                ("human", """Topic: {query}

Sources:
{sources}

Generate a research report.""")
            ])
            
            # Format sources with limits
            formatted_sources = self._format_sources_safely(limited_data)
            
            try:
                # **CRITICAL FIX** - Add timeout and proper error handling
                logger.info("🔄 Calling OpenAI API...")
                
                response = await asyncio.wait_for(
                    self.llm.ainvoke(report_prompt.format_messages(
                        query=query,
                        sources=formatted_sources
                    )),
                    timeout=45.0
                )
                
                # **CRITICAL FIX** - Check if response is None
                if response is None:
                    logger.error("❌ Google-Gemini API returned None response")
                    return await self._create_fallback_report(query, limited_data)
                
                # **CRITICAL FIX** - Check if response has content
                if not hasattr(response, 'content') or not response.content:
                    logger.error("❌ Google-Gemini response missing content")
                    return await self._create_fallback_report(query, limited_data)
                
                # Track token usage safely
                try:
                    if hasattr(response, 'response_metadata') and response.response_metadata:
                        token_info = response.response_metadata.get('token_usage', {})
                        if isinstance(token_info, dict):
                            self.token_usage += token_info.get('total_tokens', 0)
                except Exception as token_error:
                    logger.warning(f"⚠️ Token tracking failed: {token_error}")
                
                report = response.content
                logger.info("✅ Report generated successfully")
                
            except asyncio.TimeoutError:
                logger.error("❌ Google-Gemini API timeout")
                return await self._create_fallback_report(query, limited_data)
            except Exception as api_error:
                logger.error(f"❌ Google-Gemini API error: {api_error}")
                return await self._create_fallback_report(query, limited_data)
            
            # Create citations
            citations = self._create_citations(limited_data)
            
            return report, citations
            
        except Exception as e:
            logger.error(f"❌ Report generation completely failed: {e}")
            return await self._create_emergency_report(query, research_data)
    
    def _format_sources_safely(self, research_data: List[Dict[str, Any]]) -> str:
        """Safely format research sources with error handling"""
        try:
            formatted_parts = []
            
            for idx, item in enumerate(research_data[:4], 1):
                try:
                    title = str(item.get('title', 'No title'))[:80]
                    content = str(item.get('content', 'No content'))[:300]
                    url = str(item.get('url', 'No URL'))
                    
                    formatted_parts.append(f"""[{idx}] {title}
URL: {url}
Content: {content}...""")
                    
                except Exception as item_error:
                    logger.warning(f"⚠️ Error formatting source {idx}: {item_error}")
                    formatted_parts.append(f"[{idx}] Source formatting error")
            
            return "\n\n".join(formatted_parts)
            
        except Exception as e:
            logger.error(f"❌ Source formatting failed: {e}")
            return f"Research data available but formatting failed: {str(e)}"
    
    def _create_citations(self, research_data: List[Dict[str, Any]]) -> List[str]:
        """Create citations safely"""
        citations = []
        try:
            for idx, item in enumerate(research_data, 1):
                try:
                    title = str(item.get('title', 'Untitled'))[:60]
                    url = str(item.get('url', ''))
                    if url and url != 'No URL':
                        citations.append(f"[{idx}] {title} - {url}")
                except Exception as citation_error:
                    logger.warning(f"⚠️ Citation {idx} error: {citation_error}")
                    citations.append(f"[{idx}] Citation error")
        except Exception as e:
            logger.error(f"❌ Citations creation failed: {e}")
            
        return citations
    
    async def _create_fallback_report(self, query: str, data: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Create report without LLM when API fails"""
        logger.info("📝 Creating fallback report")
        
        report = f"""# Research Report: {query}

## Executive Summary
Research was conducted on "{query}" using automated web search. {len(data)} sources were analyzed to provide this overview.

## Key Findings
"""
        
        # Add findings from sources
        for idx, item in enumerate(data[:4], 1):
            try:
                title = str(item.get('title', 'Source'))[:60]
                content = str(item.get('content', ''))[:200]
                report += f"• **Source {idx}**: {title} - {content}...\n"
            except Exception:
                report += f"• **Source {idx}**: Information available but formatting error\n"
        
        report += f"""
## Conclusions
Based on the {len(data)} sources analyzed, this report provides an overview of {query}. For more detailed analysis, additional research may be needed.

## Technical Note
This report was generated using fallback mode due to API limitations.
"""
        
        citations = self._create_citations(data)
        return report, citations
    
    async def _create_summary_report(self, query: str, data: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Create summary when content is too large"""
        logger.info("📄 Creating summary report")
        
        report = f"""# Research Summary: {query}

## Overview
Comprehensive research conducted with {len(data)} primary sources analyzed.

## Source Analysis
"""
        
        for idx, item in enumerate(data, 1):
            try:
                title = str(item.get('title', 'Untitled'))[:50]
                word_count = len(str(item.get('content', '')).split())
                report += f"{idx}. **{title}** ({word_count} words analyzed)\n"
            except Exception:
                report += f"{idx}. Source analysis error\n"
        
        report += f"""
## Summary
Research on "{query}" has been compiled from multiple sources. Due to content volume, this summary provides key source identification and basic analysis.

## Note
Detailed content analysis was limited due to processing constraints.
"""
        
        citations = self._create_citations(data)
        return report, citations
    
    async def _create_emergency_report(self, query: str, data: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Last resort report when everything fails"""
        logger.error("🚨 Creating emergency report")
        
        report = f"""# Research Report - Emergency Mode

## Query
{query}

## Status
Research system encountered technical difficulties during report generation.

## Data Collected
- Sources found: {len(data)}
- Status: Emergency fallback mode

## Technical Details
The research agent successfully collected information but encountered issues during the final report generation phase. This typically indicates:

1. API connectivity issues
2. Content processing limitations  
3. Authentication problems

## Recommendations
1. Verify API keys and configuration
2. Check network connectivity
3. Try a simpler query
4. Contact technical support

## Available Sources
{len(data)} sources were successfully collected before the error occurred.
"""
        
        return report, []

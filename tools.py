# tools.py
import asyncio
import aiohttp
from typing import List, Dict, Any
from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchRun
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from config import Config

logger = logging.getLogger(__name__)

class ResearchTools:
    def __init__(self):
        self.tavily_search = TavilySearch(
            api_key=Config.TAVILY_API_KEY,
            max_results=Config.MAX_SEARCH_RESULTS,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True
        )
        self.ddg_search = DuckDuckGoSearchRun()
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Research Agent 1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def web_search(self, query: str, search_type: str = "comprehensive") -> List[Dict[str, Any]]:
        """Perform web search with multiple search engines for comprehensive results"""
        try:
            results = []
            
            # Tavily search for high-quality results
            try:
                tavily_results = await asyncio.to_thread(self.tavily_search.invoke, query)
                for result in tavily_results.get("results", []):
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "source": "tavily",
                        "score": result.get("score", 0.5),
                        "published_date": result.get("published_date"),
                    })
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}")
            
            # DuckDuckGo search for additional coverage
            try:
                ddg_results = await asyncio.to_thread(self.ddg_search.run, query)
                # Parse DDG results and add to results list
                # Implementation depends on DDG response format
            except Exception as e:
                logger.warning(f"DuckDuckGo search failed: {e}")
            
            return results[:Config.MAX_SEARCH_RESULTS]
            
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            return []
    
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape content from a URL with error handling"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Extract text content
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    return {
                        "url": url,
                        "title": soup.title.string if soup.title else "",
                        "content": text[:5000],  # Limit content length
                        "status": "success",
                        "word_count": len(text.split())
                    }
                else:
                    return {
                        "url": url,
                        "status": "error",
                        "error": f"HTTP {response.status}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {
                "url": url,
                "status": "error",
                "error": str(e)
            }
    
    async def batch_scrape(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently"""
        tasks = [self.scrape_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Scraping task failed: {result}")
            else:
                processed_results.append(result)
        
        return processed_results

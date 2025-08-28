
import asyncio
import aiohttp
from typing import List, Dict, Any
from langchain_community.tools import DuckDuckGoSearchRun
from bs4 import BeautifulSoup
import logging
from config import Config

logger = logging.getLogger(__name__)

class ResearchTools:
    def __init__(self):
        self.tavily_search = None
        self.ddg_search = DuckDuckGoSearchRun()
        self._session = None
        
        # Initialize Tavily if available
        if Config.TAVILY_API_KEY:
            try:
                from langchain_tavily import TavilySearch
                self.tavily_search = TavilySearch(
                    api_key=Config.TAVILY_API_KEY,
                    max_results=Config.MAX_SEARCH_RESULTS,
                    search_depth="basic",  # Changed to basic for faster results
                    include_answer=True,
                    include_raw_content=False  # Reduced complexity
                )
                logger.info("✅ Tavily search initialized")
            except Exception as e:
                logger.warning(f"⚠️ Tavily initialization failed: {e}")
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),  # Reduced timeout
                headers={'User-Agent': 'Research Agent 1.0'},
                connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
            )
        return self._session
    
    async def close_session(self):
        """Properly close the session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform web search with limited results"""
        try:
            results = []
            
            # Try Tavily first (limited results for speed)
            if self.tavily_search:
                try:
                    tavily_results = await asyncio.to_thread(self.tavily_search.invoke, query)
                    for result in tavily_results.get("results", [])[:max_results]:
                        results.append({
                            "title": result.get("title", "")[:100],  # Truncate titles
                            "url": result.get("url", ""),
                            "content": result.get("content", "")[:500],  # Limit content
                            "source": "tavily",
                            "score": result.get("score", 0.5),
                        })
                    logger.info(f"✅ Tavily returned {len(results)} results")
                except Exception as e:
                    logger.warning(f"⚠️ Tavily search failed: {e}")
            
            # Fallback to DuckDuckGo if needed
            if len(results) < 3:
                try:
                    ddg_results = await asyncio.to_thread(self.ddg_search.run, query)
                    if isinstance(ddg_results, str) and ddg_results.strip():
                        results.append({
                            "title": f"Search results for: {query}",
                            "url": "https://duckduckgo.com",
                            "content": ddg_results[:300],  # Limited content
                            "source": "duckduckgo",
                            "score": 0.3,
                        })
                except Exception as e:
                    logger.warning(f"⚠️ DuckDuckGo search failed: {e}")
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"❌ Web search failed: {e}")
            return [{
                "title": f"Basic info about {query}",
                "url": "https://example.com",
                "content": f"Research topic: {query}. Limited results due to technical issues.",
                "source": "fallback",
                "score": 0.1,
            }]
    
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape URL with proper session handling"""
        try:
            session = await self.get_session()
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Extract text (simplified)
                    for script in soup(["script", "style", "nav", "footer"]):
                        script.decompose()
                    
                    text = soup.get_text()
                    # Clean and limit text
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    clean_text = ' '.join(lines)[:1000]  # Strict limit
                    
                    return {
                        "url": url,
                        "title": soup.title.string[:100] if soup.title else "No title",
                        "content": clean_text,
                        "status": "success",
                        "word_count": len(clean_text.split())
                    }
                else:
                    logger.warning(f"⚠️ HTTP {response.status} for {url}")
                    return {"url": url, "status": "error", "error": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Failed to scrape {url}: {e}")
            return {"url": url, "status": "error", "error": str(e)}
    
    async def batch_scrape(self, urls: List[str], max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Scrape multiple URLs with concurrency control"""
        if not urls:
            return []
        
        # Limit URLs to prevent overwhelming
        limited_urls = urls[:max_concurrent]
        logger.info(f"🔄 Scraping {len(limited_urls)} URLs")
        
        tasks = [self.scrape_url(url) for url in limited_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"❌ Scraping task failed: {result}")
            elif isinstance(result, dict):
                processed_results.append(result)
        
        return processed_results
    
    # Context manager support
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()

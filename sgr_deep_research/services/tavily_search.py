import asyncio
import logging
from typing import Optional

from tavily import TavilyClient

from sgr_deep_research.core.models import SourceData
from sgr_deep_research.settings import get_config

logger = logging.getLogger(__name__)


class TavilySearchService:
    def __init__(self):
        config = get_config()
        self._client = TavilyClient(api_key=config.tavily.api_key, api_base_url=config.tavily.api_base_url)
        self._config = config

    @staticmethod
    def rearrange_sources(sources: list[SourceData], starting_number=1) -> list[SourceData]:
        for i, source in enumerate(sources, starting_number):
            source.number = i
        return sources

    def search(
        self,
        query: str,
        max_results: int | None = None,
        include_raw_content: bool = True,
    ) -> list[SourceData]:
        """Perform search through Tavily API and return results with
        SourceData.

        Args:
            query: Search query
            max_results: Maximum number of results (default from config)
            include_raw_content: Include raw page content

        Returns:
            List of SourceData
        """
        max_results = max_results or self._config.search.max_results
        logger.info(f"🔍 Tavily search: '{query}' (max_results={max_results})")

        try:
            # Execute search through Tavily with timeout
            response = asyncio.run(self._search_with_timeout(
                query=query,
                max_results=max_results,
                include_raw_content=include_raw_content,
            ))
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Search timeout for query: '{query}'")
            # Fallback to basic search without content scraping
            response = self._client.search(
                query=query,
                max_results=max_results,
                include_raw_content=False,
            )
        except Exception as e:
            logger.error(f"❌ Search error for query '{query}': {e}")
            # Fallback to basic search
            response = self._client.search(
                query=query,
                max_results=max_results,
                include_raw_content=False,
            )

        # Convert results to SourceData
        sources = self._convert_to_source_data(response)

        return sources

    async def _search_with_timeout(
        self,
        query: str,
        max_results: int,
        include_raw_content: bool,
    ) -> dict:
        """Execute search with timeout."""
        timeout = self._config.scraping.timeout
        
        async def _search():
            # Run the synchronous search in a thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._client.search(
                    query=query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                )
            )
        
        return await asyncio.wait_for(_search(), timeout=timeout)

    def _convert_to_source_data(self, response: dict) -> list[SourceData]:
        """Convert Tavily response to SourceData list."""
        sources = []

        for i, result in enumerate(response.get("results", [])):
            if not result.get("url", ""):
                continue

            source = SourceData(
                number=i,
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=result.get("content", ""),
            )
            
            # Handle full content with fallback
            if result.get("raw_content", ""):
                source.full_content = result["raw_content"]
                source.char_count = len(source.full_content)
            elif self._config.scraping.fallback_to_snippets:
                # Use snippet as fallback content
                source.full_content = result.get("content", "")
                source.char_count = len(source.full_content)
                logger.info(f"📄 Using snippet as fallback content for: {source.url}")
            
            sources.append(source)
        return sources

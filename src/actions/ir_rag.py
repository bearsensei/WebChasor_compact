"""
Information Retrieval and Reasoning Action for WebChasor
Handles factual queries that require external information retrieval.
"""

import os
import json
import asyncio
import logging
import requests
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, replace
from enum import Enum
import re
import hashlib
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import time
from utils.timectx import TimeContext, parse_time_intent
from artifacts import Action, Context, Artifact
from actions.serpapi_search import SerpAPISearch
from actions.google_search import GoogleCustomSearch
from actions.gcp_vertex_search import GCPVertexSearch
from actions.querymaker import QueryMaker
from config_manager import get_config
# Import components from standalone modules
from planner import Planner, PlanTask, ExtractionPlan
from actions.visitor import WebVisitor, WebPage
from actions.ranker import ContentRanker, ContentPassage
from actions.extractor import InformationExtractor, ExtractedVariable

# Setup logging
logger = logging.getLogger(__name__)

class RetrievalProvider(Enum):
    """Available retrieval providers"""
    SERPAPI = "serpapi"
    GOOGLE_CUSTOM_SEARCH = "google_custom_search"
    GCP_VERTEX_SEARCH = "gcp_vertex_search"
    VECTOR_DB = "vector_db"  # Placeholder for future implementation
    HYBRID = "hybrid"        # Combine both sources

# PlanTask and ExtractionPlan now imported from planner module

# WebPage, ContentPassage, ExtractedVariable now imported from standalone modules

@dataclass
class SearchResult:
    """Structured search result"""
    title: str
    snippet: str
    url: str
    source: str
    result_type: str = "organic"  # organic, answer_box, knowledge_graph, news
    position: int = 0
    date: Optional[str] = None
    confidence: float = 0.0

@dataclass
class IRConfig:
    """Configuration for IR_RAG execution"""
    # Search configuration
    max_search_results: int = 10
    search_provider: RetrievalProvider = RetrievalProvider.SERPAPI
    search_providers: List[RetrievalProvider] = None  # Multiple providers for hybrid/fallback
    search_location: str = "Hong Kong"
    search_language: str = "zh-cn"
    search_gl: str = None  # Geo-location country code (e.g., 'hk', 'us', 'cn')
    search_time_filter: str = None  # Time-based search filter (e.g., 'qdr:d', 'qdr:w', 'qdr:y')
    
    # Content fetching
    enable_web_scraping: bool = False  # 🔧 NEW: Web scraping toggle
    max_pages_to_visit: int = 5
    fetch_timeout: int = 10
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_passage_length: int = 20  # Added missing field
    
    # Extraction configuration
    confidence_threshold: float = 0.7  # Renamed to match config
    max_passages_per_task: int = 3
    max_extraction_attempts: int = 3  # Added missing field
    
    # Vector DB configuration (placeholder)
    vector_db_enabled: bool = False
    vector_top_k: int = 10
    hybrid_weight: float = 0.5  # Weight for combining vector and search results

# Planner, WebVisitor, ContentRanker, InformationExtractor now imported from standalone modules

class IR_RAG(Action):
    """
    Information Retrieval and Reasoning Action
    
    Handles queries that require external information retrieval and synthesis.
    Workflow: Planning → Retrieval → Reading → Ranking → Extraction → Synthesis
    """
    
    name = "IR_RAG"
    requires_tools = ["search", "synthesizer"]
    max_time_s = 120
    max_tokens_in = 10000
    max_tokens_out = 2000
    
    def __init__(self, config: Optional[IRConfig] = None, llm_client=None):
        """Initialize IR_RAG with configuration and LLM client"""
        self.config = config or self._load_config_from_yaml()
        self.llm_client = llm_client
        
        # Initialize components with config-driven model names
        cfg = get_config()
        planner_model = cfg.get('models.planner.model_name', 'gpt-4')
        extractor_model = cfg.get('models.extractor.model_name', 'gpt-3.5-turbo')
        
        self.planner = Planner(llm_client, planner_model)
        self.querymaker = QueryMaker()
        
        # Initialize search tool(s) based on provider configuration
        # Support multiple providers for hybrid search
        if self.config.search_providers:
            # Multiple providers configured
            self.search_tools = []
            provider_names = []
            print(f"[IR_RAG][INIT] search_providers configured: {[p.value for p in self.config.search_providers]}")
            for provider in self.config.search_providers:
                if provider == RetrievalProvider.GOOGLE_CUSTOM_SEARCH:
                    self.search_tools.append(GoogleCustomSearch())
                    provider_names.append("Google Custom Search")
                elif provider == RetrievalProvider.GCP_VERTEX_SEARCH:
                    self.search_tools.append(GCPVertexSearch())
                    provider_names.append("GCP Vertex AI Search")
                elif provider == RetrievalProvider.SERPAPI:
                    self.search_tools.append(SerpAPISearch())
                    provider_names.append("SerpAPI")
            
            # Use first tool as primary (for backward compatibility)
            self.search_tool = self.search_tools[0] if self.search_tools else SerpAPISearch()
            
            print(f"[IR_RAG][INIT] Configured {len(self.search_tools)} search providers: {', '.join(provider_names)}")
            if cfg.is_decision_logging_enabled('ir_rag'):
                print(f"[IR_RAG][INIT] Using multiple search providers: {', '.join(provider_names)}")
        else:
            # Single provider configured (backward compatibility)
            self.search_tools = []
            if self.config.search_provider == RetrievalProvider.GOOGLE_CUSTOM_SEARCH:
                self.search_tool = GoogleCustomSearch()
                if cfg.is_decision_logging_enabled('ir_rag'):
                    print(f"[IR_RAG][INIT] Using Google Custom Search API")
            elif self.config.search_provider == RetrievalProvider.GCP_VERTEX_SEARCH:
                self.search_tool = GCPVertexSearch()
                if cfg.is_decision_logging_enabled('ir_rag'):
                    print(f"[IR_RAG][INIT] Using GCP Vertex AI Search")
            else:
                self.search_tool = SerpAPISearch()
                if cfg.is_decision_logging_enabled('ir_rag'):
                    print(f"[IR_RAG][INIT] Using SerpAPI")
        
        self.visitor = WebVisitor(self.config)
        self.ranker = ContentRanker(self.config)
        self.extractor = InformationExtractor(llm_client, extractor_model)
        
        if cfg.is_decision_logging_enabled('ir_rag'):
            print(f"[IR_RAG][INIT] provider={self.config.search_provider.value} max_pages={self.config.max_pages_to_visit} web_scraping={'ON' if self.config.enable_web_scraping else 'OFF'}")
    
    def _load_config_from_yaml(self) -> IRConfig:
        """Load IR_RAG configuration from YAML config file"""
        cfg = get_config()
        ir_config = cfg.get_section('ir_rag')
        
        # Parse provider(s) - support both single and multiple providers
        provider_str = ir_config.get('search', {}).get('provider', 'serpapi')
        print(f"[IR_RAG][CONFIG] Raw provider string: '{provider_str}'")
        search_providers = None
        search_provider = RetrievalProvider.SERPAPI
        
        if ',' in provider_str:
            # Multiple providers specified (e.g., "serpapi,gcp_vertex_search")
            provider_list = [p.strip() for p in provider_str.split(',')]
            print(f"[IR_RAG][CONFIG] Detected multiple providers: {provider_list}")
            search_providers = []
            for p in provider_list:
                try:
                    search_providers.append(RetrievalProvider(p))
                    print(f"[IR_RAG][CONFIG] Successfully parsed provider: {p}")
                except ValueError:
                    print(f"[IR_RAG][WARN] Unknown provider '{p}', skipping")
            # Use first provider as primary
            if search_providers:
                search_provider = search_providers[0]
                print(f"[IR_RAG][CONFIG] Primary provider: {search_provider.value}")
        else:
            # Single provider
            print(f"[IR_RAG][CONFIG] Detected single provider: {provider_str}")
            try:
                search_provider = RetrievalProvider(provider_str)
            except ValueError:
                print(f"[IR_RAG][WARN] Unknown provider '{provider_str}', using serpapi")
                search_provider = RetrievalProvider.SERPAPI
        
        return IRConfig(
            search_provider=search_provider,
            search_providers=search_providers,
            max_search_results=ir_config.get('search', {}).get('max_results', 10),
            max_pages_to_visit=ir_config.get('web_scraping', {}).get('max_pages', 5),
            search_location=ir_config.get('search', {}).get('location', 'Hong Kong'),
            search_language=ir_config.get('search', {}).get('language', 'zh-cn'),
            search_gl=ir_config.get('search', {}).get('gl', None),  # Read gl from config
            search_time_filter=ir_config.get('search', {}).get('time_filter', None),  # Read time filter from config
            enable_web_scraping=ir_config.get('web_scraping', {}).get('enabled', False),
            chunk_size=ir_config.get('content', {}).get('chunk_size', 500),
            chunk_overlap=ir_config.get('content', {}).get('chunk_overlap', 50),
            max_passages_per_task=ir_config.get('content', {}).get('max_passages_per_task', 3),
            min_passage_length=ir_config.get('content', {}).get('min_passage_length', 20),
            confidence_threshold=ir_config.get('extraction', {}).get('confidence_threshold', 0.7),
            max_extraction_attempts=ir_config.get('extraction', {}).get('max_extraction_attempts', 3)
        )
    
    async def run(self, ctx: Context, toolset) -> Artifact:
        """
        Execute the complete IR workflow
        
        Args:
            ctx: Context containing query and metadata
            toolset: Available tools including synthesizer
            
        Returns:
            Artifact: Synthesized information with citations
        """
        try:
            print(f"[IR_RAG][START] query={ctx.query[:100]}...")
            
            # Step 1: Planning - Create extraction plan
            plan = await self.planner.plan(ctx.query)
            print(f"[IR_RAG][PLAN] tasks={len(plan.tasks_to_extract)}")
            
            # Step 2: Retrieval - Search for information
            search_results = await self._retrieve_information(ctx.query, plan, ctx)
            print(f"[IR_RAG][SEARCH] results={search_results}")
            print(f"[IR_RAG][SEARCH] results={len(search_results)}")
            
            if not search_results:
                return self._create_no_results_artifact(ctx.query)
            
            # Step 3: Reading - Hybrid approach: snippets + selective web scraping
            # Strategy: Use ALL snippets as primary source, supplement with a few high-quality pages
            
            # 3a. Convert ALL search snippets to "virtual pages" for extraction
            snippet_pages = self._create_pages_from_snippets(search_results)
            print(f"[IR_RAG][SNIPPET] Created {len(snippet_pages)} virtual pages from snippets")
            
            # 3b. Optionally fetch a few high-quality web pages for deep content
            web_pages = []
            if self.config.enable_web_scraping:
                print(f"[IR_RAG][WEB] scraping=ENABLED, fetching selective pages...")
                selected_urls = await self._select_seed_urls(search_results, plan, ctx.query)
                # Use config max_pages setting
                max_fetch = min(self.config.max_pages_to_visit, len(selected_urls))
                selected_urls = selected_urls[:max_fetch]
                print(f"[IR_RAG][WEB] will fetch {len(selected_urls)} pages (config max_pages={self.config.max_pages_to_visit})")
                web_pages = await self.visitor.fetch_many(selected_urls)
                print(f"[IR_RAG][WEB] fetched={len(web_pages)} supplementary pages")
            else:
                print(f"[IR_RAG][WEB] scraping=DISABLED, using snippets only")
            
            # 3c. Combine snippet pages + web pages for comprehensive coverage
            all_pages = snippet_pages + web_pages
            print(f"[IR_RAG][CONTENT] Total pages for extraction: {len(all_pages)} (snippets: {len(snippet_pages)}, web: {len(web_pages)})")
            
            # Step 4: Ranking - Rank content passages by relevance
            ranked_passages = self.ranker.rank_passages(plan, all_pages)
            
            # Step 5: Extraction - Extract structured information
            extracted_vars = await self.extractor.extract_variables(plan, ranked_passages, search_results)
            print(f"[IR_RAG][EXTRACT] variables={len(extracted_vars)}")
            
            # Step 6: Synthesis - Generate final response
            artifact = await self._synthesize_response(ctx, plan, extracted_vars, search_results, toolset)
            
            print(f"[IR_RAG][DONE] workflow completed")
            return artifact
            
        except Exception as e:
            logger.error(f"IR_RAG workflow failed: {e}")
            return self._create_error_artifact(ctx.query, str(e))
    
    async def _retrieve_information(self, query: str, plan: ExtractionPlan, ctx: Context) -> List[SearchResult]:
        """Retrieve information using configured search provider"""
        
        if self.config.search_provider in (RetrievalProvider.SERPAPI, RetrievalProvider.GOOGLE_CUSTOM_SEARCH, RetrievalProvider.GCP_VERTEX_SEARCH):
            return await self._web_search(query, plan, ctx)
        elif self.config.search_provider == RetrievalProvider.VECTOR_DB:
            return await self._vector_search(query, plan)
        elif self.config.search_provider == RetrievalProvider.HYBRID:
            return await self._hybrid_search(query, plan, ctx)
        else:
            raise ValueError(f"Unsupported search provider: {self.config.search_provider}")
    
    async def _web_search(self, query: str, plan: ExtractionPlan, ctx: Context) -> List[SearchResult]:
        """Search using web search API (SerpAPI, Google Custom Search, or GCP Vertex Search) with multiple diverse queries (batch mode)"""
        provider_map = {
            RetrievalProvider.GOOGLE_CUSTOM_SEARCH: "Google Custom Search",
            RetrievalProvider.GCP_VERTEX_SEARCH: "GCP Vertex AI Search",
            RetrievalProvider.SERPAPI: "SerpAPI"
        }
        
        # Check if using multiple providers
        if self.config.search_providers and len(self.config.search_providers) > 1:
            provider_names = [provider_map.get(p, str(p)) for p in self.config.search_providers]
            print(f"🔍 IR_RAG: Searching with multiple providers: {', '.join(provider_names)}...")
            return await self._multi_provider_search(query, plan, ctx)
        else:
            provider_name = provider_map.get(self.config.search_provider, "SerpAPI")
            print(f"🔍 IR_RAG: Searching with {provider_name}...")

        # Build multiple search queries from plan
        search_queries = self._build_search_query(query, plan, ctx)
        
        print(f"🔍 IR_RAG: Generated {len(search_queries)} search queries:")
        for i, q in enumerate(search_queries, 1):
            print(f"  {i}. {q}")
        print(f"🔍 IR_RAG: Time intent: {ctx.time_context.intent if ctx.time_context else 'None'}")
        
        # Use batch_call_flat to search all queries concurrently
        try:
            print(f"🔍 IR_RAG: Executing batch search with {len(search_queries)} queries...")
            
            # Get config values for batch search
            cfg = get_config()
            concurrent = cfg.get('ir_rag.search.concurrent', 8)
            qps = cfg.get('ir_rag.search.qps', 5)
            retries = cfg.get('ir_rag.search.retries', 2)
            
            # Execute batch search (synchronous but internally concurrent)
            raw_results = self.search_tool.batch_call_flat(
                queries=search_queries,
                num_results=self.config.max_search_results,
                location=self.config.search_location,
                language=self.config.search_language,
                gl=self.config.search_gl,  # Pass geo-location code
                concurrent=concurrent,
                qps=qps,
                retries=retries,
                tbs=self.config.search_time_filter  # Use time filter from config
            )
            
            print(f"🔍 IR_RAG: Batch search returned {len(raw_results)} total results")
            
            # Convert raw results to SearchResult objects and deduplicate by URL
            all_search_results = []
            seen_urls = set()
            
            for result in raw_results:
                url = result.get('link', '')
                
                # Skip if we've already seen this URL
                if url in seen_urls:
                    continue
                
                seen_urls.add(url)
                
                search_result = SearchResult(
                    title=result.get('title', ''),
                    snippet=result.get('snippet', ''),
                    url=url,
                    source=result.get('source', ''),
                    result_type=result.get('type', 'organic'),
                    position=result.get('position', 0),
                    date=result.get('date'),
                    confidence=0.8  # Default confidence for SerpAPI results
                )
                all_search_results.append(search_result)
            
            print(f"🔍 IR_RAG: Retrieved {len(all_search_results)} unique results from {len(search_queries)} queries")
            return all_search_results
            
        except Exception as e:
            logger.error(f"[IR_RAG][SEARCH] Batch search failed: {e}")
            return []
    
    async def _multi_provider_search(self, query: str, plan: ExtractionPlan, ctx: Context) -> List[SearchResult]:
        """Search using multiple providers and merge results"""
        print(f"🔍 IR_RAG: Executing multi-provider search with {len(self.search_tools)} providers...")
        
        # Build search queries
        search_queries = self._build_search_query(query, plan, ctx)
        print(f"🔍 IR_RAG: Generated {len(search_queries)} search queries for each provider")
        
        # Get config values for batch search
        cfg = get_config()
        concurrent = cfg.get('ir_rag.search.concurrent', 8)
        qps = cfg.get('ir_rag.search.qps', 5)
        retries = cfg.get('ir_rag.search.retries', 2)
        
        all_results = []
        provider_stats = {}
        
        # Execute search with each provider
        for i, search_tool in enumerate(self.search_tools):
            provider_name = self.config.search_providers[i].value if i < len(self.config.search_providers) else f"Provider {i+1}"
            
            try:
                print(f"🔍 IR_RAG: Searching with {provider_name}...")
                
                # Execute batch search
                raw_results = search_tool.batch_call_flat(
                    queries=search_queries,
                    num_results=self.config.max_search_results,
                    location=self.config.search_location,
                    language=self.config.search_language,
                    gl=self.config.search_gl,  # Pass geo-location code
                    concurrent=concurrent,
                    qps=qps,
                    retries=retries,
                    tbs=self.config.search_time_filter  # Use time filter from config
                )
                
                provider_stats[provider_name] = len(raw_results)
                print(f"🔍 IR_RAG: {provider_name} returned {len(raw_results)} results")
                
                # Convert to SearchResult objects
                for result in raw_results:
                    search_result = SearchResult(
                        title=result.get('title', ''),
                        snippet=result.get('snippet', ''),
                        url=result.get('link', ''),
                        source=result.get('source', ''),
                        result_type=result.get('type', 'organic'),
                        position=result.get('position', 0),
                        date=result.get('date'),
                        confidence=0.8
                    )
                    all_results.append(search_result)
                    
            except Exception as e:
                logger.error(f"[IR_RAG][SEARCH] {provider_name} failed: {e}")
                provider_stats[provider_name] = 0
                # Continue with other providers even if one fails
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        print(f"🔍 IR_RAG: Multi-provider search stats: {provider_stats}")
        print(f"🔍 IR_RAG: Total results: {len(all_results)}, Unique results: {len(unique_results)}")
        
        return unique_results
    
    async def _vector_search(self, query: str, plan: ExtractionPlan) -> List[SearchResult]:
        """Search using vector database (placeholder)"""
        print(f"🔍 IR_RAG: Vector search not implemented yet, using fallback...")
        
        # Placeholder for future vector database integration
        # This would query a vector database with embeddings
        
        return []
    
    async def _hybrid_search(self, query: str, plan: ExtractionPlan, ctx: Context) -> List[SearchResult]:
        """Combine web search and vector search results"""
        print(f"🔍 IR_RAG: Hybrid search not fully implemented, using web search only...")
        
        # For now, just use web search
        # In the future, this would combine results from both sources
        return await self._web_search(query, plan, ctx)
    
    def _build_search_query(self, original_query: str, plan: ExtractionPlan, ctx: Context) -> List[str]:
        """
        Build optimized search queries using LLM-based QueryMaker.
        
        Args:
            original_query: User's original query
            plan: Extraction plan from planner
            ctx: Context with time and other information
            
        Returns:
            List of diverse search queries
        """
        try:
            # Use QueryMaker to generate diverse queries
            queries = self.querymaker.generate_queries(original_query, ctx)
            logger.info(f"[IR_RAG][QUERY] Generated {len(queries)} search queries")
            return queries
        except Exception as e:
            logger.error(f"[IR_RAG][QUERY] QueryMaker failed: {e}, using fallback")
            # Fallback to original query
            return [original_query]
    
    def _parse_serpapi_results(self, raw_results: str, query: str) -> List[SearchResult]:
        """Parse SerpAPI formatted results into structured format"""
        # This is a simplified parser - in production you'd want more robust parsing
        results = []
        
        if "No results found" in raw_results or "Error:" in raw_results:
            return results
        
        # Extract results using regex patterns
        # This is a basic implementation - you might want to modify SerpAPISearch 
        # to return structured data instead
        
        # Look for title and URL patterns
        url_pattern = r'📎 Link: (https?://[^\s\n]+)'
        title_pattern = r'\*\*(.*?)\*\*'
        
        urls = re.findall(url_pattern, raw_results)
        titles = re.findall(title_pattern, raw_results)
        
        # Create search results
        for i, (title, url) in enumerate(zip(titles, urls)):
            if url and title:
                results.append(SearchResult(
                    title=title,
                    snippet="",  # Would need more parsing to extract snippets
                    url=url,
                    source=urlparse(url).netloc,
                    position=i + 1,
                    confidence=0.8
                ))
        
        print(f"🔍 IR_RAG: Parsed {len(results)} structured results from SerpAPI")
        return results
    
    async def _select_seed_urls(self, search_results: List[SearchResult], plan: ExtractionPlan, user_query: str) -> List[str]:
        """
        Use LLM to intelligently select the most relevant URLs to visit.
        This replaces complex heuristic-based logic with semantic understanding.
        """
        if not search_results:
            return []

        max_targets = max(1, getattr(self.config, "max_pages_to_visit", 5))
        
        # Prepare candidates for LLM
        candidates = []
        for i, result in enumerate(search_results[:30], 1):  # Limit to top 30 to avoid token limits
            candidates.append({
                "index": i,
                "title": result.title or "No title",
                "url": result.url,
                "snippet": (result.snippet or "")[:300],  # Truncate long snippets
                "source": result.source or "",
                "position": result.position
            })
        
        # Extract query context
        entity_name = plan.entity if plan and plan.entity else ""
        
        print(f"[SELECTOR][LLM] Calling LLM to select {max_targets} URLs from {len(candidates)} candidates")
        print(f"[SELECTOR][LLM] Query: '{user_query}', Entity: '{entity_name}'")
        
        # Build LLM prompt
        prompt = self._build_url_selection_prompt(user_query, entity_name, candidates, max_targets)
        
        try:
            # Call LLM
            response = await self._call_llm_for_url_selection(prompt)
            selected_indices = self._parse_url_selection_response(response)
            
            # Map indices to URLs
            selected_urls = []
            for idx in selected_indices[:max_targets]:
                if 1 <= idx <= len(search_results):
                    url = search_results[idx - 1].url
                    selected_urls.append(url)
                    print(f"🌐 SELECTOR: [LLM_SELECTED] {url}")
            
            if selected_urls:
                print(f"[SELECTOR][LLM] Successfully selected {len(selected_urls)} URLs")
                return selected_urls
            else:
                print(f"[SELECTOR][LLM] No valid URLs selected, falling back to simple strategy")
                
        except Exception as e:
            print(f"[SELECTOR][ERROR] LLM selection failed: {e}, falling back to simple strategy")
        
        # Fallback: simple position-based selection
        print(f"[SELECTOR][FALLBACK] Using position-based selection")
        selected_urls = [r.url for r in search_results[:max_targets]]
        for url in selected_urls:
            print(f"🌐 SELECTOR: [FALLBACK] {url}")
        
        return selected_urls
    
    def _build_url_selection_prompt(self, user_query: str, entity_name: str, candidates: List[Dict], max_targets: int) -> str:
        """Build prompt for LLM to select most relevant URLs."""
        candidates_text = "\n".join([
            f"{c['index']}. [{c['source']}] {c['title']}\n   URL: {c['url']}\n   Snippet: {c['snippet'][:200]}..."
            for c in candidates[:30]  # Limit to avoid token overflow
        ])
        
        entity_context = f"The user is asking about: {entity_name}" if entity_name else ""
        
        prompt = f"""You are a web search expert. Given a user query and search results, select the {max_targets} most relevant and authoritative URLs to visit.

User Query: {user_query}
{entity_context}

Selection Criteria (in order of priority):
1. **Direct Relevance**: URLs that directly answer the query
2. **Authority**: Official sources, employer websites, personal homepages > news articles > general pages
   - For person queries: personal homepage, employer profile page, official bio
   - For company queries: official website, about page
   - Wikipedia is good but should not replace primary sources
3. **Freshness**: Prefer recent content when relevant
4. **Diversity**: Include different types of sources (not all news articles)
5. **Disambiguation**: If there are multiple entities with the same name, prefer the most relevant one based on query context

Search Results:
{candidates_text}

Task:
Select exactly {max_targets} URLs (by index number) that would provide the most comprehensive and accurate information.

Response Format (JSON only, no explanation):
{{
  "selected_indices": [1, 3, 7, ...],
  "reasoning": "Brief explanation of selection strategy"
}}
"""
        return prompt
    
    async def _call_llm_for_url_selection(self, prompt: str) -> str:
        """Call LLM API to select URLs."""
        if not self.llm_client:
            raise ValueError("LLM client not available")
        
        # Use same model as extractor for consistency
        cfg = get_config()
        model_name = cfg.get('models.extractor.model_name', 'gpt-oss-20b')
        temperature = cfg.get('models.extractor.temperature', 0.0)
        max_tokens = cfg.get('models.extractor.max_tokens', 1000)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a precise web search expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content.strip()
            return content
            
        except Exception as e:
            print(f"[SELECTOR][LLM] API call failed: {e}")
            raise
    
    def _parse_url_selection_response(self, response: str) -> List[int]:
        """Parse LLM response to extract selected indices."""
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*"selected_indices"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                indices = data.get("selected_indices", [])
                print(f"[SELECTOR][LLM] Selected indices: {indices}")
                print(f"[SELECTOR][LLM] Reasoning: {data.get('reasoning', 'N/A')}")
                return [int(idx) for idx in indices if isinstance(idx, (int, str)) and str(idx).isdigit()]
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[SELECTOR][LLM] JSON parse error: {e}")
        
        # Fallback: try to extract numbers from response
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            indices = [int(n) for n in numbers[:10]]  # Take first 10 numbers found
            print(f"[SELECTOR][LLM] Extracted indices from text: {indices}")
            return indices
        
        print(f"[SELECTOR][LLM] Could not parse response: {response[:200]}")
        return []

    
    def _create_pages_from_snippets(self, search_results: List[SearchResult]) -> List[WebPage]:
        """Create dummy WebPage objects from search results if web scraping is disabled"""
        print(f"🌐 IR_RAG: Creating dummy pages from search snippets for ranking...")
        pages = []
        for i, result in enumerate(search_results):
            pages.append(WebPage(
                url=result.url,
                title=result.title,
                content=result.snippet or "",
                headings=[],
                metadata={
                    'status_code': 200,
                    'content_type': 'text/html',
                    'source_type': 'snippet',
                    'pseudo_source': 'SERP_SNIPPET',
                    'original_url': result.url,
                    'title': result.title,
                    'snippet': result.snippet,
                    'serp_position': result.position,
                    'snippet_date': result.date,
                    'snippet_confidence': result.confidence,
                },
                fetch_time=0.0,
                success=True
            ))
        return pages

    async def _synthesize_response(self, ctx: Context, plan: ExtractionPlan, 
                                 extracted_vars: Dict[str, ExtractedVariable], 
                                 search_results: List[SearchResult], toolset) -> Artifact:
        """Synthesize final response using extracted information"""
        
        # Build materials for synthesis
        materials = self._build_synthesis_materials(plan, extracted_vars, search_results)
        
        # Get current time information
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_date = datetime.datetime.now().strftime("%Y年%m月%d日")
        
        # Build time context string
        time_info = f"Current time: {current_time} ({current_date})"
        if ctx.time_context:
            time_info += f"\nTime intent: {ctx.time_context.intent}"
            if ctx.time_context.window and ctx.time_context.window[1]:
                time_info += f"\nTime window: {ctx.time_context.window[0]} to {ctx.time_context.window[1]}"
        
        # Create synthesis prompt
        synthesis_prompt = f"""
        Based on the extracted information, provide a comprehensive answer to: {ctx.query}. You MUST use all retrieved information, no missing information. Be as comprehensive as possible. Add background information if there is any possible.
        
        **IMPORTANT TIME CONTEXT**:
        {time_info}
        
        Use the following extracted information:
        {materials}
        
        Please provide a well-structured response with:
        1. Clear sections based on the information found
        2. DO NOT include any citation numbers like [1], [2], etc. in the main text
        3. Use diverse formatting: include bullet points, numbered lists, comparison tables where appropriate
        4. For key facts/statistics/comparisons, use tables or lists instead of long paragraphs
        5. Mix paragraph text with structured formats (lists, tables) for better readability
        6. Factual accuracy based on the sources
        7. **Time awareness**: The current time is {current_time}. Filter and present information based on this current time. Distinguish between past, present, and future events clearly.
        8. Include a "References" or "参考来源" section at the very end listing all sources
        9. MUST use all retrieved information, no missing information.
        """
        

        # Call synthesizer using the new generate method for proper global prompt support
        if hasattr(toolset, 'synthesizer'):
            # Set up constraints for IR_RAG synthesis with temperature from config
            cfg = get_config()
            ir_rag_temperature = cfg.get('models.synthesizer.temperature', 0.5)
            
            constraints = {
                "language": "auto",  # Let synthesizer detect language
                "tone": "factual, authoritative",
                "temperature": ir_rag_temperature,  # Use config value
                "instruction_hint": f"Provide a comprehensive answer with clear sections. Use diverse formatting: bullet points, numbered lists, and tables where appropriate. Mix paragraphs with structured formats for better readability. DO NOT use citation numbers [1], [2], etc. in the main text. Only include a reference list at the end.\n\n**CRITICAL**: The current time is {current_time} ({current_date}). Be aware of this when discussing events, dates, and timelines. Clearly distinguish between past, present, and future."
            }
            
            response = await toolset.synthesizer.generate(
                category="INFORMATION_RETRIEVAL",
                style_key="auto",  # 使用 auto 启用自动语言检测和样式选择
                constraints=constraints,
                materials=f"Query: {ctx.query}\n\n**Current Time**: {current_time} ({current_date})\n\nExtracted Information:\n{materials}",
                task_scaffold=None
            )
        else:
            response = self._create_fallback_response(ctx.query, extracted_vars)
        # Build citations
        citations = self._build_citations(extracted_vars, search_results)
        
        # Create artifact
        artifact = Artifact(
            kind="text",
            content=response,
            meta={
                'query': ctx.query,
                'plan': plan,
                'extracted_variables': {k: v.value for k, v in extracted_vars.items()},
                'citations': citations,
                'search_results_count': len(search_results),
                'extraction_confidence': self._calculate_overall_confidence(extracted_vars),
                'sources': list(set([url for var in extracted_vars.values() for url in var.provenance]))
            }
        )
        
        return artifact
    
    def _build_synthesis_materials(self, plan: ExtractionPlan, 
                                 extracted_vars: Dict[str, ExtractedVariable], 
                                 search_results: List[SearchResult]) -> str:
        """Build materials string for synthesis"""
        materials = []
        citation_counter = 1
        
        # Add extracted variables with numbered citations
        for var_name, var_data in extracted_vars.items():
            if var_data.value and var_data.confidence > 0.3:
                # Use numbered citation instead of variable name
                materials.append(f"[{citation_counter}] **{var_name}**: {var_data.value}")
                citation_counter += 1
        
        # Add ALL search snippets (not just top 3) for comprehensive coverage
        # This ensures we use the rich snippet information from all search results
        for result in search_results[:20]:  # Use top 10 snippets
            if result.snippet:
                materials.append(f"[{citation_counter}] {result.title}: {result.snippet}")
                citation_counter += 1
        
        return "\n\n".join(materials)
    
    def _build_citations(self, extracted_vars: Dict[str, ExtractedVariable], 
                        search_results: List[SearchResult]) -> List[Dict[str, str]]:
        """Build citation list"""
        citations = []
        seen_urls = set()
        
        # Add sources from extracted variables
        for var in extracted_vars.values():
            for url in var.provenance:
                if url not in seen_urls:
                    # Find corresponding search result for title
                    title = url
                    for result in search_results:
                        if result.url == url:
                            title = result.title
                            break
                    
                    citations.append({
                        'title': title,
                        'url': url,
                        'source': urlparse(url).netloc
                    })
                    seen_urls.add(url)
        
        return citations
    
    def _calculate_overall_confidence(self, extracted_vars: Dict[str, ExtractedVariable]) -> float:
        """Calculate overall confidence score"""
        if not extracted_vars:
            return 0.0
        
        confidences = [var.confidence for var in extracted_vars.values()]
        return sum(confidences) / len(confidences)
    
    def _create_fallback_response(self, query: str, extracted_vars: Dict[str, ExtractedVariable]) -> str:
        """Create fallback response when synthesizer is not available"""
        response_parts = [f"Based on the available information, here's what I found about: {query}\n"]
        
        for var_name, var_data in extracted_vars.items():
            if var_data.value:
                response_parts.append(f"• **{var_name.replace('_', ' ').title()}**: {var_data.value}")
        
        if not any(var.value for var in extracted_vars.values()):
            response_parts.append("Unfortunately, I couldn't find specific information to answer your query.")
        
        return "\n".join(response_parts)
    
    def _create_no_results_artifact(self, query: str) -> Artifact:
        """Create artifact when no search results are found"""
        return Artifact(
            kind="text",
            content=f"I couldn't find any relevant information for your query: {query}. Please try rephrasing your question or checking for typos.",
            meta={
                'query': query,
                'error': 'no_search_results',
                'search_results_count': 0
            }
        )
    
    def _create_fetch_failed_artifact(self, query: str, search_results: List[SearchResult]) -> Artifact:
        """Create artifact when page fetching fails"""
        return Artifact(
            kind="text",
            content=f"I found search results for your query but couldn't access the web pages. Here are the search results I found:\n\n" +
                   "\n".join([f"• {result.title}: {result.url}" for result in search_results[:5]]),
            meta={
                'query': query,
                'error': 'fetch_failed',
                'search_results_count': len(search_results)
            }
        )
    
    def _create_error_artifact(self, query: str, error: str) -> Artifact:
        """Create artifact for general errors"""
        return Artifact(
            kind="text",
            content=f"I encountered an error while processing your query: {query}. Error: {error}",
            meta={
                'query': query,
                'error': error
            }
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of IR_RAG components"""
        health = {
            'status': 'healthy',
            'components': {},
            'config': {
                'search_provider': self.config.search_provider.value,
                'max_pages': self.config.max_pages_to_visit,
                'vector_db_enabled': self.config.vector_db_enabled
            }
        }
        
        # Check SerpAPI
        try:
            if self.search_tool.api_key:
                health['components']['serpapi'] = 'configured'
            else:
                health['components']['serpapi'] = 'missing_api_key'
        except Exception as e:
            health['components']['serpapi'] = f'error: {e}'
        
        # Check LLM client
        if self.llm_client:
            health['components']['llm'] = 'available'
        else:
            health['components']['llm'] = 'not_configured'
        
        return health

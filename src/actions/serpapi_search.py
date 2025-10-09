"""
SerpAPI Search Tool for WebChaser
Provides Google search results through SerpAPI
"""

import os
import json
import requests
from typing import List, Dict, Any, Union
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from config_manager import get_config


class SerpAPISearch:
    """SerpAPI search tool for WebChaser"""
    
    name = "serpapi_search"
    description = "Search using SerpAPI (Google Search API alternative)"
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': 'Search query string',
            'required': True
        },
        {
            'name': 'num_results',
            'type': 'integer',
            'description': 'Number of results to return (default: 10, max: 100)',
            'required': False
        },
        {
            'name': 'location',
            'type': 'string',
            'description': 'Location for localized search (e.g., "Hong Kong", "United States")',
            'required': False
        },
        {
            'name': 'language',
            'type': 'string',
            'description': 'Language code (e.g., "zh-cn", "en")',
            'required': False
        }
    ]

    def __init__(self):
        # API key still from environment (sensitive credential)
        self.api_key = os.getenv('SERPAPI_KEY', '')
        if not self.api_key:
            print("Warning: SERPAPI_KEY environment variable not set")
        
        self.base_url = "https://serpapi.com/search"

        # Load batch & concurrency config from config.yaml
        cfg = get_config()
        self.qps = cfg.get('ir_rag.search.qps', 5)               # Max requests per second to SerpAPI
        self.concurrent = cfg.get('ir_rag.search.concurrent', 8) # Max concurrent requests
        self.retries = cfg.get('ir_rag.search.retries', 2)       # Per-request retries
        self.timeout_s = cfg.get('ir_rag.web_scraping.timeout', 30)  # Per-request timeout seconds

        # Shared state for simple client-side rate limiting across threads
        self._rl_lock = threading.Lock()
        self._last_call_ts = 0.0
        self._min_interval = 1.0 / max(1, self.qps)
    
    def _rate_limit(self):
        """
        Simple client-side QPS limiter shared across threads.
        Ensures at most `self.qps` requests per second.
        """
        with self._rl_lock:
            now = time.monotonic()
            wait = (self._last_call_ts + self._min_interval) - now
            if wait > 0:
                time.sleep(wait)
            self._last_call_ts = time.monotonic()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """Execute search and return formatted results"""
        try:
            # Parse parameters
            if isinstance(params, str):
                # Simple string query
                query = params
                num_results = kwargs.get('num_results', 10)
                location = kwargs.get('location', 'Hong Kong')
                language = kwargs.get('language', 'zh-cn')
                engine = kwargs.get('engine', 'google')
            else:
                # Dictionary parameters
                query = params.get('query', '')
                num_results = params.get('num_results', 10)
                location = params.get('location', 'Hong Kong')
                language = params.get('language', 'zh-cn')
                engine = params.get('engine', 'google')
            
            if not query:
                return "Error: Query parameter is required"
            
            # Execute search
            results = self._execute_search(query, num_results, location, language, engine)
            
            # Format results for display
            return self._format_results(results, query)
            
        except Exception as e:
            print(f"SerpAPI Search Error: {e}")
            return f"SerpAPI Search Error: {str(e)}"

    def get_structured_results(self, query: str, num_results: int = 10, location: str = 'Hong Kong', 
                             language: str = 'zh-cn', engine: str = 'google', tbs: str = None, gl: str = None) -> List[Dict[str, Any]]:
        """Get structured search results for IR_RAG integration"""
        try:
            if not self.api_key:
                print("Error: SERPAPI_KEY not configured")
                return []
            
            return self._execute_search(query, num_results, location, language, engine, tbs, gl)
        except Exception as e:
            print(f"SerpAPI Structured Search Error: {e}")
            return []

    def _execute_search(self, query: str, num_results: int, location: str, language: str, engine: str = 'google', tbs: str = None, gl: str = None) -> List[Dict[str, Any]]:
        """Execute the actual SerpAPI search"""
        
        # Auto-infer gl (geo-location) if not provided
        if gl is None:
            # Location to country code mapping
            location_gl_map = {
                'hong kong': 'hk',
                'china': 'cn',
                'singapore': 'sg',
                'japan': 'jp',
                'korea': 'kr',
                'taiwan': 'tw',
                'united kingdom': 'uk',
                'united states': 'us',
                'canada': 'ca',
                'australia': 'au'
            }
            loc_lower = location.lower()
            gl = next((code for loc, code in location_gl_map.items() if loc in loc_lower), 'us')
        
        # Prepare search parameters
        search_params = {
            'engine': engine,
            'q': query,
            'api_key': self.api_key,
            'num': min(num_results, 100),  # SerpAPI max is 100
            'location': location,
            'hl': language,
            'gl': gl,
        }
        
        # Add time-based filtering if provided
        if tbs:
            search_params['tbs'] = tbs
        
        print(f"SerpAPI Search: '{query}' (engine: {engine}, location: {location}, results: {num_results})")
        
        try:
            response = requests.get(self.base_url, params=search_params, timeout=self.timeout_s)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'error' in data:
                print(f"SerpAPI Error: {data['error']}")
                return []
            
            # Extract results based on engine type
            organic_results = data.get('organic_results', [])
            news_results = data.get('news_results', [])
            
            # Also get answer box if available
            answer_box = data.get('answer_box', {})
            knowledge_graph = data.get('knowledge_graph', {})
            
            
            # Process results
            processed_results = []
            print(f"Raw API Response Keys: {list(data.keys())}")
            if 'answer_box' in data:
                print(f"Answer Box: {data['answer_box']}")
            if 'knowledge_graph' in data:
                print(f"Knowledge Graph: {data['knowledge_graph']}")
            # Add answer box first if available
            if answer_box:
                # Extract snippet from various Answer Box formats
                snippet = answer_box.get('snippet', '') or answer_box.get('description', '')
                
                # For organic_result type Answer Box (e.g., broker data)
                if answer_box.get('type') == 'organic_result' and answer_box.get('contents'):
                    contents = answer_box.get('contents', {})
                    table = contents.get('table', [])
                    if table:
                        # Format table data into readable text
                        table_text = []
                        for row in table:
                            if isinstance(row, list) and len(row) >= 2:
                                table_text.append(f"{row[0]}: {row[1]}")
                        snippet = snippet + "\n" + "\n".join(table_text) if snippet else "\n".join(table_text)
                
                # For finance_results type Answer Box (Google Finance)
                elif answer_box.get('type') == 'finance_results':
                    price = answer_box.get('price')
                    table = answer_box.get('table', [])
                    price_movement = answer_box.get('price_movement', {})
                    
                    finance_text = []
                    if price:
                        finance_text.append(f"当前价格: {price} {answer_box.get('currency', '')}")
                    if price_movement:
                        finance_text.append(f"涨跌: {price_movement.get('movement', '')} {price_movement.get('price', '')} ({price_movement.get('percentage', '')}%)")
                        finance_text.append(f"日期: {price_movement.get('date', '')}")
                    for item in table:
                        if isinstance(item, dict):
                            finance_text.append(f"{item.get('name', '')}: {item.get('value', '')}")
                    
                    snippet = snippet + "\n" + "\n".join(finance_text) if snippet else "\n".join(finance_text)
                
                processed_results.append({
                    'type': 'answer_box',
                    'title': answer_box.get('title', ''),
                    'snippet': snippet,
                    'link': answer_box.get('link', ''),
                    'source': answer_box.get('displayed_link', '') or answer_box.get('source', '')
                })
            
            # Add knowledge graph if available
            if knowledge_graph:
                # Extract description
                snippet = knowledge_graph.get('description', '')
                
                # Metadata fields to skip (not content)
                skip_keys = {'type', 'kgmid', 'knowledge_graph_search_link', 'serpapi_knowledge_graph_search_link',
                            'title', 'description', 'website', 'source', 'entity_type'}
                
                # Dynamically extract all structured fields
                structured_fields = []
                for key, value in knowledge_graph.items():
                    # Skip metadata and already extracted fields
                    if key in skip_keys:
                        continue
                    
                    # Extract string values
                    if isinstance(value, str) and value.strip():
                        structured_fields.append(f"{key}: {value}")
                    # Extract list values (e.g., founders, subsidiaries)
                    elif isinstance(value, list) and value:
                        if all(isinstance(item, str) for item in value[:3]):  # Check first 3 items
                            structured_fields.append(f"{key}: {', '.join(value[:5])}")  # Limit to 5 items
                
                # Append structured data to snippet
                if structured_fields:
                    snippet = snippet + "\n\n" + "\n".join(structured_fields[:10])  # Limit to 10 fields
                
                processed_results.append({
                    'type': 'knowledge_graph',
                    'title': knowledge_graph.get('title', ''),
                    'snippet': snippet,
                    'link': knowledge_graph.get('website', ''),
                    'source': knowledge_graph.get('source', {}).get('name', '')
                })
            
            # Add news results (for google_news engine)
            if news_results and engine == 'google_news':
                for result in news_results[:num_results]:
                    processed_results.append({
                        'type': 'news',
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'link': result.get('link', ''),
                        'source': result.get('source', ''),
                        'date': result.get('date', ''),
                        'position': result.get('position', 0)
                    })
            
            # Add organic results (for regular google engine)
            if organic_results and engine != 'google_news':
                for result in organic_results[:num_results]:
                    processed_results.append({
                        'type': 'organic',
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'link': result.get('link', ''),
                        'source': result.get('displayed_link', ''),
                        'position': result.get('position', 0)
                    })
            
            print(f"SerpAPI returned {len(processed_results)} results")
            print(f"Processed Results: {processed_results}")
            return processed_results
            
        except requests.RequestException as e:
            print(f"SerpAPI Request Error: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"SerpAPI JSON Error: {e}")
            return []

    def _format_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results for display"""
        
        if not results:
            return f"No results found for: {query}"
        
        formatted = f"**SerpAPI Search Results for:** {query}\n\n"
        
        for i, result in enumerate(results, 1):
            result_type = result.get('type', 'organic')
            title = result.get('title', 'No title')
            snippet = result.get('snippet', 'No description available')
            link = result.get('link', '')
            source = result.get('source', '')
            
            # Add type indicator
            if result_type == 'answer_box':
                formatted += "**Answer Box:**\n"
            elif result_type == 'knowledge_graph':
                formatted += "**Knowledge Graph:**\n"
            elif result_type == 'news':
                formatted += f"{i}. "
            else:
                formatted += f"{i}. "
            
            # Add title
            if link:
                formatted += f"{title}\n"
            else:
                formatted += f"{title}\n"
            
            # Add snippet
            if snippet:
                # Clean and truncate snippet
                clean_snippet = snippet.replace('\n', ' ').strip()
                if len(clean_snippet) > 200:
                    clean_snippet = clean_snippet[:200] + "..."
                formatted += f"{clean_snippet}\n"
            
            # Add date for news results
            if result_type == 'news' and result.get('date'):
                formatted += f"Date: {result.get('date')}\n"
            
            # Add source and link
            if source and link:
                formatted += f"Source: {source}\n"
                formatted += f"Link: {link}\n"
            elif link:
                formatted += f"Link: {link}\n"
            
            formatted += "\n"
        
        return formatted

    def _execute_with_retry(self, query: str, num_results: int, location: str, language: str, engine: str, tbs: str = None, gl: str = None) -> List[Dict[str, Any]]:
        """
        Execute a single SerpAPI search with retries, backoff, and client-side rate limiting.
        """
        attempt = 0
        backoff = 0.6
        while True:
            try:
                self._rate_limit()
                return self._execute_search(query, num_results, location, language, engine, tbs, gl)
            except Exception as e:
                # _execute_search already catches RequestException/JSON errors and returns []
                # If we reach here due to an unexpected exception, treat as failure to retry.
                if attempt >= self.retries:
                    print(f"[SerpAPI] Retry exhausted for query='{query}': {e}")
                    return []
                sleep_s = backoff * (1.0 + random.random() * 0.25)
                time.sleep(sleep_s)
                backoff *= 2
                attempt += 1

    def batch_call(
        self,
        queries: List[str],
        num_results: int = 10,
        location: str = 'Hong Kong',
        language: str = 'zh-cn',
        engine: str = 'google',
        tbs: str = None,
        gl: str = None,
        concurrent: int = None,
        qps: int = None,
        retries: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch search for multiple queries (SerpAPI does not support true server-side batching).
        This method performs client-side concurrent requests with QPS limiting and retries.

        Returns a mapping: { query: [result_dict, ...] }
        """
        if not isinstance(queries, list) or not queries:
            return {}

        # Override runtime knobs if provided
        if concurrent is not None: self.concurrent = int(concurrent)
        if qps is not None:
            self.qps = int(qps)
            self._min_interval = 1.0 / max(1, self.qps)
        if retries is not None: self.retries = int(retries)

        out: Dict[str, List[Dict[str, Any]]] = {}

        def worker(q: str) -> None:
            try:
                res = self._execute_with_retry(q, num_results, location, language, engine, tbs, gl)
                out[q] = res
            except Exception as e:
                print(f"[SerpAPI] Worker error for '{q}': {e}")
                out[q] = []

        # Thread pool with bounded concurrency
        with ThreadPoolExecutor(max_workers=self.concurrent) as ex:
            futs = {ex.submit(worker, q): q for q in queries}
            for fut in as_completed(futs):
                _ = futs[fut]  # ensure exceptions are surfaced
                try:
                    fut.result()
                except Exception as e:
                    q = futs[fut]
                    print(f"[SerpAPI] Future error for '{q}': {e}")
                    out[q] = out.get(q, [])

        return out

    def batch_call_flat(
        self,
        queries: List[str],
        num_results: int = 10,
        location: str = 'Hong Kong',
        language: str = 'zh-cn',
        engine: str = 'google',
        tbs: str = None,
        gl: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch search and flatten all results into a single list with `query` annotated.
        """
        mapping = self.batch_call(queries, num_results, location, language, engine, tbs, gl, **kwargs)
        flat: List[Dict[str, Any]] = []
        for q, items in mapping.items():
            for it in items:
                it2 = dict(it)  # shallow copy
                it2["query"] = q
                flat.append(it2)
        return flat

# Test function
def test_serpapi():
    """Test SerpAPI functionality"""
    search_tool = SerpAPISearch()

    # Single query test (existing behavior)
    test_query = "香港天气"
    result = search_tool.call({'query': test_query, 'num_results': 5})
    print("=" * 60)
    print("SerpAPI Test Results (single):")
    print("=" * 60)
    print(result)

    # Batch queries test (new)
    test_queries = ["香港天气", "香港股市 指数", "香港大学 校长 任期"]
    mapping = search_tool.batch_call(test_queries, num_results=5, location="Hong Kong", language="zh-cn")
    print("=" * 60)
    print("SerpAPI Test Results (batch):")
    print("=" * 60)
    for q, items in mapping.items():
        print(f"[{q}] -> {len(items)} items")
        if items[:1]:
            print(f"  First title: {items[0].get('title')}")

if __name__ == "__main__":
    test_serpapi()
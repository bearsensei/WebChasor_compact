"""
Google Custom Search API Tool for WebChaser
Provides Google search results through official Google Custom Search API
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


class GoogleCustomSearch:
    """Google Custom Search API tool for WebChaser"""
    
    name = "google_custom_search"
    description = "Search using Google Custom Search API (Official Google API)"
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
            'description': 'Number of results to return (default: 10, max: 10 per request)',
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
        # API credentials from environment (sensitive credentials)
        self.api_key = os.getenv('GOOGLE_SEARCH_KEY', '')
        self.cse_id = os.getenv('GOOGLE_CSE_ID', '')
        
        if not self.api_key:
            print("Warning: GOOGLE_SEARCH_KEY environment variable not set")
        if not self.cse_id:
            print("Warning: GOOGLE_CSE_ID environment variable not set")
        
        self.base_url = "https://www.googleapis.com/customsearch/v1"

        # Load batch & concurrency config from config.yaml
        cfg = get_config()
        self.qps = cfg.get('ir_rag.search.qps', 5)               # Max requests per second
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
            else:
                # Dictionary parameters
                query = params.get('query', '')
                num_results = params.get('num_results', 10)
                location = params.get('location', 'Hong Kong')
                language = params.get('language', 'zh-cn')
            
            if not query:
                return "Error: Query parameter is required"
            
            # Execute search
            results = self._execute_search(query, num_results, location, language)
            
            # Format results for display
            return self._format_results(results, query)
            
        except Exception as e:
            print(f"Google Custom Search Error: {e}")
            return f"Google Custom Search Error: {str(e)}"

    def get_structured_results(self, query: str, num_results: int = 10, location: str = 'Hong Kong', 
                             language: str = 'zh-cn', tbs: str = None) -> List[Dict[str, Any]]:
        """Get structured search results for IR_RAG integration"""
        try:
            if not self.api_key or not self.cse_id:
                print("Error: GOOGLE_SEARCH_KEY or GOOGLE_CSE_ID not configured")
                return []
            
            return self._execute_search(query, num_results, location, language, tbs)
        except Exception as e:
            print(f"Google Custom Search Structured Search Error: {e}")
            return []

    def _execute_search(self, query: str, num_results: int, location: str, language: str, tbs: str = None) -> List[Dict[str, Any]]:
        """Execute the actual Google Custom Search API search"""
        
        # Google Custom Search API limits to 10 results per request
        # For more results, we need to paginate
        num_results = min(num_results, 10)
        
        # Map location to country code (gl parameter)
        gl_map = {
            'hong kong': 'hk',
            'united states': 'us',
            'china': 'cn',
            'japan': 'jp',
            'united kingdom': 'uk',
            'germany': 'de',
            'france': 'fr',
        }
        gl = gl_map.get(location.lower(), 'us')
        
        # Map language to hl parameter (interface language)
        hl_map = {
            'zh-cn': 'zh-CN',
            'zh-tw': 'zh-TW',
            'en': 'en',
            'ja': 'ja',
            'de': 'de',
            'fr': 'fr',
        }
        hl = hl_map.get(language.lower(), 'en')
        
        # Prepare search parameters
        search_params = {
            'key': self.api_key,
            'cx': self.cse_id,
            'q': query,
            'num': num_results,
            'gl': gl,
            'hl': hl,
            'safe': 'active',  # Enable safe search
        }
        
        # Add time-based filtering if provided
        if tbs:
            search_params['sort'] = f'date:r:{tbs}'  # Google CSE uses 'sort' instead of 'tbs'
        
        print(f"[GoogleSearch] Searching: '{query}' (location: {location}/{gl}, language: {language}/{hl}, results: {num_results})")
        
        try:
            response = requests.get(self.base_url, params=search_params, timeout=self.timeout_s)
            response.raise_for_status()
            
            data = response.json()
            
            # Debug: Print search information
            search_info = data.get('searchInformation', {})
            total_results = search_info.get('totalResults', '0')
            print(f"[GoogleSearch][DEBUG] Total results available: {total_results}")
            
            # Check for API errors
            if 'error' in data:
                error_msg = data['error'].get('message', 'Unknown error')
                error_code = data['error'].get('code', '')
                print(f"[GoogleSearch][ERROR] API Error ({error_code}): {error_msg}")
                
                # Provide helpful message for quota errors
                if error_code in (403, 429) or 'quota' in error_msg.lower() or 'limit' in error_msg.lower():
                    print(f"[GoogleSearch][HINT] Google Custom Search has strict rate limits (1 QPS, 100/day free).")
                    print(f"[GoogleSearch][HINT] Consider switching to 'serpapi' in config.yaml for better limits.")
                
                return []
            
            # Extract results
            items = data.get('items', [])
            
            # Debug: If no items but totalResults > 0, there's a configuration issue
            if not items and int(total_results) > 0:
                print(f"[GoogleSearch][WARNING] Search found {total_results} results but returned 0 items!")
                print(f"[GoogleSearch][HINT] This usually means your Custom Search Engine is not configured to 'Search the entire web'.")
                print(f"[GoogleSearch][HINT] Visit https://programmablesearchengine.google.com/ and enable 'Search the entire web' in your CSE settings.")
            elif not items:
                print(f"[GoogleSearch][INFO] No results found for query: '{query}'")
                print(f"[GoogleSearch][HINT] Try a different search term or check if the topic exists online.")
            
            # Process results
            processed_results = []
            
            for i, item in enumerate(items, 1):
                result = {
                    'type': 'organic',
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'link': item.get('link', ''),
                    'source': item.get('displayLink', ''),
                    'position': i
                }
                
                # Add page map data if available (rich snippets)
                if 'pagemap' in item:
                    pagemap = item['pagemap']
                    
                    # Check for meta tags
                    if 'metatags' in pagemap and pagemap['metatags']:
                        metatags = pagemap['metatags'][0]
                        result['meta'] = {
                            'og_title': metatags.get('og:title'),
                            'og_description': metatags.get('og:description'),
                            'og_image': metatags.get('og:image'),
                            'author': metatags.get('author'),
                            'date': metatags.get('article:published_time') or metatags.get('date'),
                        }
                    
                    # Check for thumbnails
                    if 'cse_thumbnail' in pagemap:
                        result['thumbnail'] = pagemap['cse_thumbnail'][0].get('src')
                
                processed_results.append(result)
            
            print(f"[GoogleSearch] Returned {len(processed_results)} results")
            return processed_results
            
        except requests.RequestException as e:
            print(f"Google Custom Search Request Error: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Google Custom Search JSON Error: {e}")
            return []

    def _format_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results for display"""
        
        if not results:
            return f"No results found for: {query}"
        
        formatted = f"**Google Custom Search Results for:** {query}\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            snippet = result.get('snippet', 'No description available')
            link = result.get('link', '')
            source = result.get('source', '')
            
            formatted += f"{i}. {title}\n"
            
            # Add snippet
            if snippet:
                # Clean and truncate snippet
                clean_snippet = snippet.replace('\n', ' ').strip()
                if len(clean_snippet) > 200:
                    clean_snippet = clean_snippet[:200] + "..."
                formatted += f"{clean_snippet}\n"
            
            # Add metadata if available
            if 'meta' in result and result['meta'].get('date'):
                formatted += f"Date: {result['meta']['date']}\n"
            
            # Add source and link
            if source and link:
                formatted += f"Source: {source}\n"
                formatted += f"Link: {link}\n"
            elif link:
                formatted += f"Link: {link}\n"
            
            formatted += "\n"
        
        return formatted

    def _execute_with_retry(self, query: str, num_results: int, location: str, language: str, tbs: str = None) -> List[Dict[str, Any]]:
        """
        Execute a single Google Custom Search API search with retries, backoff, and client-side rate limiting.
        """
        attempt = 0
        backoff = 2.0  # Increased initial backoff for Google API rate limits
        while True:
            try:
                self._rate_limit()
                return self._execute_search(query, num_results, location, language, tbs)
            except Exception as e:
                error_msg = str(e)
                # Check for rate limit errors (429)
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    if attempt >= self.retries:
                        print(f"[GoogleSearch] Rate limit exceeded for query='{query}', giving up after {attempt} retries")
                        return []
                    # Longer backoff for rate limit errors
                    sleep_s = backoff * 2 * (1.0 + random.random() * 0.5)
                    print(f"[GoogleSearch] Rate limit hit, waiting {sleep_s:.1f}s before retry {attempt+1}/{self.retries}")
                    time.sleep(sleep_s)
                    backoff *= 2
                    attempt += 1
                else:
                    # Other errors
                    if attempt >= self.retries:
                        print(f"[GoogleSearch] Retry exhausted for query='{query}': {e}")
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
        tbs: str = None,
        concurrent: int = None,
        qps: int = None,
        retries: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch search for multiple queries.
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
                res = self._execute_with_retry(q, num_results, location, language, tbs)
                out[q] = res
            except Exception as e:
                print(f"[GoogleSearch] Worker error for '{q}': {e}")
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
                    print(f"[GoogleSearch] Future error for '{q}': {e}")
                    out[q] = out.get(q, [])

        return out

    def batch_call_flat(
        self,
        queries: List[str],
        num_results: int = 10,
        location: str = 'Hong Kong',
        language: str = 'zh-cn',
        tbs: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch search and flatten all results into a single list with `query` annotated.
        """
        mapping = self.batch_call(queries, num_results, location, language, tbs, **kwargs)
        flat: List[Dict[str, Any]] = []
        for q, items in mapping.items():
            for it in items:
                it2 = dict(it)  # shallow copy
                it2["query"] = q
                flat.append(it2)
        return flat


# Test function
def test_google_search():
    """Test Google Custom Search functionality"""
    search_tool = GoogleCustomSearch()

    # Single query test
    test_query = "香港天气"
    result = search_tool.call({'query': test_query, 'num_results': 5})
    print("=" * 60)
    print("Google Custom Search Test Results (single):")
    print("=" * 60)
    print(result)

    # Batch queries test
    test_queries = ["香港天气", "香港股市 指数", "香港大学 校长 任期"]
    mapping = search_tool.batch_call(test_queries, num_results=5, location="Hong Kong", language="zh-cn")
    print("=" * 60)
    print("Google Custom Search Test Results (batch):")
    print("=" * 60)
    for q, items in mapping.items():
        print(f"[{q}] -> {len(items)} items")
        if items[:1]:
            print(f"  First title: {items[0].get('title')}")


if __name__ == "__main__":
    test_google_search()


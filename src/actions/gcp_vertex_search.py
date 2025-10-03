"""
GCP Vertex AI Search Tool for WebChaser
Provides search results through Google Cloud Vertex AI Search (Discovery Engine)
"""

import os
import json
import requests
from typing import List, Dict, Any, Union, Optional
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from config_manager import get_config

try:
    from google.api_core.client_options import ClientOptions
    from google.cloud import discoveryengine_v1 as discoveryengine
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    print("Warning: Google Cloud libraries not available. Install with: pip install google-cloud-discoveryengine")


class GCPVertexSearch:
    """GCP Vertex AI Search tool for WebChaser"""
    
    name = "gcp_vertex_search"
    description = "Search using Google Cloud Vertex AI Search (Discovery Engine)"
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
            'description': 'Number of results to return (default: 10)',
            'required': False
        },
        {
            'name': 'location',
            'type': 'string',
            'description': 'Location for localized search (not used by GCP Vertex)',
            'required': False
        },
        {
            'name': 'language',
            'type': 'string',
            'description': 'Language code (not used by GCP Vertex)',
            'required': False
        }
    ]

    def __init__(self):
        # API credentials from environment (sensitive credentials)
        self.project_id = os.getenv('GCP_PROJECT_ID', '')
        self.location = os.getenv('GCP_LOCATION', 'global')
        self.engine_id = os.getenv('GCP_ENGINE_ID', '')
        self.api_key = os.getenv('GCP_API_KEY', '')
        
        if not self.project_id:
            print("Warning: GCP_PROJECT_ID environment variable not set")
        if not self.engine_id:
            print("Warning: GCP_ENGINE_ID environment variable not set")
        if not self.api_key:
            print("Warning: GCP_API_KEY environment variable not set")
        
        if not GCP_AVAILABLE:
            self.error_msg = "Google Cloud libraries not available. Install with: pip install google-cloud-discoveryengine"
        elif not all([self.project_id, self.engine_id, self.api_key]):
            self.error_msg = "Missing GCP configuration. Please set GCP_PROJECT_ID, GCP_ENGINE_ID, and GCP_API_KEY environment variables."
        else:
            self.error_msg = None

        # Load batch & concurrency config from config.yaml
        cfg = get_config()
        self.qps = cfg.get('ir_rag.search.qps', 5)
        self.concurrent = cfg.get('ir_rag.search.concurrent', 8)
        self.retries = cfg.get('ir_rag.search.retries', 2)
        self.timeout_s = cfg.get('ir_rag.web_scraping.timeout', 30)

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
        # Check availability
        if self.error_msg:
            return f"[GCP Vertex Search] {self.error_msg}"
        
        try:
            # Parse parameters
            if isinstance(params, str):
                # Simple string query
                query = params
                num_results = kwargs.get('num_results', 10)
            else:
                # Dictionary parameters
                query = params.get('query', '')
                num_results = params.get('num_results', 10)
            
            if not query:
                return "Error: Query parameter is required"
            
            # Execute search
            results = self._execute_search(query, num_results)
            
            # Format results for display
            return self._format_results(results, query)
            
        except Exception as e:
            print(f"GCP Vertex Search Error: {e}")
            return f"GCP Vertex Search Error: {str(e)}"

    def get_structured_results(self, query: str, num_results: int = 10, location: str = 'Hong Kong', 
                             language: str = 'zh-cn', tbs: str = None) -> List[Dict[str, Any]]:
        """Get structured search results for IR_RAG integration"""
        # Check availability
        if self.error_msg:
            print(f"[GCPVertexSearch] {self.error_msg}")
            return []
        
        try:
            return self._execute_search(query, num_results, tbs)
        except Exception as e:
            print(f"GCP Vertex Search Structured Search Error: {e}")
            return []

    def _execute_search(self, query: str, num_results: int, tbs: str = None) -> List[Dict[str, Any]]:
        """Execute the actual GCP Vertex AI search using Discovery Engine"""
        
        # Check if GCP libraries are available
        if not GCP_AVAILABLE:
            print(f"[GCPVertexSearch][ERROR] Google Cloud libraries not available")
            print(f"[GCPVertexSearch][HINT] Install with: pip install google-cloud-discoveryengine")
            return []
        
        # Check configuration
        if self.error_msg:
            print(f"[GCPVertexSearch][ERROR] {self.error_msg}")
            return []
        
        print(f"[GCPVertexSearch] Searching: '{query}' (results: {num_results})")
        
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud import discoveryengine_v1 as discoveryengine
            
            client_options = ClientOptions(
                api_key=self.api_key,
                api_endpoint=(
                    f"{self.location}-discoveryengine.googleapis.com"
                    if self.location != "global"
                    else None
                ),
            )

            # Create client
            client = discoveryengine.SearchServiceClient(client_options=client_options)

            # Build serving config path
            serving_config = (
                f"projects/{self.project_id}/locations/{self.location}/"
                f"collections/default_collection/engines/{self.engine_id}/"
                f"servingConfigs/default_config"
            )

            # Create search request
            request = discoveryengine.SearchRequest(
                serving_config=serving_config,
                query=query,
                page_size=num_results,
            )

            # Execute search
            page_result = client.search_lite(request)
            
            # Process results
            processed_results = []
            count = 0
            
            for response in page_result:
                if count >= num_results:
                    break
                    
                # Extract document data
                if hasattr(response, 'document') and response.document:
                    doc_data = self._extract_document_data(response.document)
                    if doc_data:
                        # Convert to standard format matching other search providers
                        result = {
                            'type': 'organic',
                            'title': doc_data.get('title', ''),
                            'snippet': doc_data.get('content', ''),
                            'link': doc_data.get('uri', ''),
                            'source': self._extract_domain(doc_data.get('uri', '')),
                            'position': count + 1
                        }
                        processed_results.append(result)
                        count += 1
            
            print(f"[GCPVertexSearch] Returned {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            print(f"GCP Vertex Search Request Error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_document_data(self, doc) -> Optional[Dict[str, Any]]:
        """Extract document data from the GCP document structure using MapComposite access"""
        try:
            doc_data = {
                'title': 'No Title',
                'content': 'No Content',
                'uri': '',
            }
            
            # Access derived_struct_data as a MapComposite
            if hasattr(doc, 'derived_struct_data') and doc.derived_struct_data:
                derived_data = doc.derived_struct_data
                
                # Extract title
                if 'title' in derived_data:
                    title_value = derived_data['title']
                    if isinstance(title_value, str):
                        doc_data['title'] = title_value
                    elif hasattr(title_value, 'string_value'):
                        doc_data['title'] = title_value.string_value
                
                # Extract snippets/content
                if 'snippets' in derived_data:
                    snippets_value = derived_data['snippets']
                    for snippet_item in snippets_value:
                        if 'snippet' in snippet_item:
                            snippet_text = snippet_item['snippet']
                            if isinstance(snippet_text, str):
                                doc_data['content'] = snippet_text
                                break
                
                # Extract link/URI
                if 'link' in derived_data:
                    link_value = derived_data['link']
                    if isinstance(link_value, str):
                        doc_data['uri'] = link_value
                    elif hasattr(link_value, 'string_value'):
                        doc_data['uri'] = link_value.string_value
            
            return doc_data
            
        except Exception as e:
            print(f"[GCPVertexSearch] Error extracting document data: {e}")
            return None

    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return ''

    def _format_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results for display"""
        
        if not results:
            return f"No results found for: {query}"
        
        formatted = f"**GCP Vertex Search Results for:** {query}\n\n"
        
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
            
            # Add source and link
            if source and link:
                formatted += f"Source: {source}\n"
                formatted += f"Link: {link}\n"
            elif link:
                formatted += f"Link: {link}\n"
            
            formatted += "\n"
        
        return formatted

    def _execute_with_retry(self, query: str, num_results: int, tbs: str = None) -> List[Dict[str, Any]]:
        """
        Execute a single GCP Vertex AI search with retries, backoff, and client-side rate limiting.
        """
        attempt = 0
        backoff = 2.0
        while True:
            try:
                self._rate_limit()
                return self._execute_search(query, num_results, tbs)
            except Exception as e:
                error_msg = str(e)
                # Check for rate limit errors
                if "429" in error_msg or "Too Many Requests" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    if attempt >= self.retries:
                        print(f"[GCPVertexSearch] Rate limit exceeded for query='{query}', giving up after {attempt} retries")
                        return []
                    sleep_s = backoff * 2 * (1.0 + random.random() * 0.5)
                    print(f"[GCPVertexSearch] Rate limit hit, waiting {sleep_s:.1f}s before retry {attempt+1}/{self.retries}")
                    time.sleep(sleep_s)
                    backoff *= 2
                    attempt += 1
                else:
                    # Other errors
                    if attempt >= self.retries:
                        print(f"[GCPVertexSearch] Retry exhausted for query='{query}': {e}")
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
                res = self._execute_with_retry(q, num_results, tbs)
                out[q] = res
            except Exception as e:
                print(f"[GCPVertexSearch] Worker error for '{q}': {e}")
                out[q] = []

        # Thread pool with bounded concurrency
        with ThreadPoolExecutor(max_workers=self.concurrent) as ex:
            futs = {ex.submit(worker, q): q for q in queries}
            for fut in as_completed(futs):
                _ = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    q = futs[fut]
                    print(f"[GCPVertexSearch] Future error for '{q}': {e}")
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
                it2 = dict(it)
                it2["query"] = q
                flat.append(it2)
        return flat


# Test function
def test_gcp_vertex_search():
    """Test GCP Vertex Search functionality"""
    
    # Check environment variables
    required_vars = ['GCP_PROJECT_ID', 'GCP_ENGINE_ID', 'GCP_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these environment variables:")
        for var in missing_vars:
            print(f"   export {var}='your_value_here'")
        return False
    
    print("All required environment variables are set")
    
    # Test the search tool
    try:
        search_tool = GCPVertexSearch()
        
        # Single query test
        test_query = "李家超活动"
        result = search_tool.call({'query': test_query, 'num_results': 5})
        print("=" * 60)
        print("GCP Vertex Search Test Results (single):")
        print("=" * 60)
        print(result)

        # Batch queries test
        test_queries = ["香港天气", "香港股市 指数", "香港大学 校长 任期"]
        mapping = search_tool.batch_call(test_queries, num_results=5)
        print("=" * 60)
        print("GCP Vertex Search Test Results (batch):")
        print("=" * 60)
        for q, items in mapping.items():
            print(f"[{q}] -> {len(items)} items")
            if items[:1]:
                print(f"  First title: {items[0].get('title')}")
        
        return True
            
    except Exception as e:
        print(f"Error testing GCP Vertex Search: {str(e)}")
        return False


if __name__ == "__main__":
    test_gcp_vertex_search()



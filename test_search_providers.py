"""
Test script to compare SerpAPI and Google Custom Search API
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from actions.serpapi_search import SerpAPISearch
from actions.google_search import GoogleCustomSearch


def test_serpapi():
    """Test SerpAPI"""
    print("=" * 80)
    print("[TEST][SerpAPI] Testing SerpAPI Search")
    print("=" * 80)
    
    search = SerpAPISearch()
    
    # Test single query
    query = "浸会大学现任校长"
    print(f"\n[TEST][Query] {query}")
    results = search.get_structured_results(query, num_results=5)
    
    if results:
        print(f"[TEST][Results] Found {len(results)} results")
        for i, result in enumerate(results[:3], 1):
            print(f"\n[{i}] {result.get('title', 'No title')}")
            print(f"    Source: {result.get('source', 'N/A')}")
            print(f"    Snippet: {result.get('snippet', 'N/A')[:100]}...")
    else:
        print("[TEST][Error] No results returned")
    
    # Test with time filter
    print(f"\n[TEST][Query with Time Filter] {query} (past month)")
    results_filtered = search.get_structured_results(query, num_results=5, tbs="qdr:m")
    
    if results_filtered:
        print(f"[TEST][Results] Found {len(results_filtered)} results")
        for i, result in enumerate(results_filtered[:2], 1):
            print(f"\n[{i}] {result.get('title', 'No title')}")
            print(f"    Source: {result.get('source', 'N/A')}")
    else:
        print("[TEST][Error] No results returned")


def test_google_custom_search():
    """Test Google Custom Search API"""
    print("\n" + "=" * 80)
    print("[TEST][GoogleSearch] Testing Google Custom Search API")
    print("=" * 80)
    
    search = GoogleCustomSearch()
    
    # Test single query
    query = "浸会大学现任校长"
    print(f"\n[TEST][Query] {query}")
    results = search.get_structured_results(query, num_results=5)
    
    if results:
        print(f"[TEST][Results] Found {len(results)} results")
        for i, result in enumerate(results[:3], 1):
            print(f"\n[{i}] {result.get('title', 'No title')}")
            print(f"    Source: {result.get('source', 'N/A')}")
            print(f"    Snippet: {result.get('snippet', 'N/A')[:100]}...")
    else:
        print("[TEST][Error] No results returned")


def test_batch_search():
    """Test batch search functionality"""
    print("\n" + "=" * 80)
    print("[TEST][Batch] Testing Batch Search")
    print("=" * 80)
    
    queries = [
        "浸会大学现任校长",
        "香港天气",
        "香港股市指数"
    ]
    
    # Test SerpAPI batch
    print("\n[TEST][SerpAPI Batch]")
    serpapi = SerpAPISearch()
    results_map = serpapi.batch_call(queries, num_results=3)
    
    for query, results in results_map.items():
        print(f"\n[Query] {query}")
        print(f"[Results] {len(results)} items")
        if results:
            print(f"[First] {results[0].get('title', 'N/A')}")
    
    # Test Google Custom Search batch
    print("\n[TEST][GoogleSearch Batch]")
    google_search = GoogleCustomSearch()
    results_map = google_search.batch_call(queries, num_results=3)
    
    for query, results in results_map.items():
        print(f"\n[Query] {query}")
        print(f"[Results] {len(results)} items")
        if results:
            print(f"[First] {results[0].get('title', 'N/A')}")


def main():
    """Main test runner"""
    print("\n" + "=" * 80)
    print("[TEST][Start] WebChasor Search Providers Test")
    print("=" * 80)
    
    # Check environment variables
    serpapi_key = os.getenv('SERPAPI_KEY')
    google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
    google_cse_id = os.getenv('GOOGLE_CSE_ID')
    
    print(f"\n[ENV] SERPAPI_KEY: {'✓ Set' if serpapi_key else '✗ Not Set'}")
    print(f"[ENV] GOOGLE_SEARCH_API_KEY: {'✓ Set' if google_api_key else '✗ Not Set'}")
    print(f"[ENV] GOOGLE_CSE_ID: {'✓ Set' if google_cse_id else '✗ Not Set'}")
    
    # Run tests based on available credentials
    if serpapi_key:
        test_serpapi()
    else:
        print("\n[SKIP] Skipping SerpAPI test (no API key)")
    
    if google_api_key and google_cse_id:
        test_google_custom_search()
    else:
        print("\n[SKIP] Skipping Google Custom Search test (no API key or CSE ID)")
    
    if serpapi_key and google_api_key and google_cse_id:
        test_batch_search()
    else:
        print("\n[SKIP] Skipping batch test (need both API credentials)")
    
    print("\n" + "=" * 80)
    print("[TEST][Complete] All tests finished")
    print("=" * 80)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Test script to compare SerpAPI and GCP Vertex Search results
"""
import os
import sys

# Add project root and src to path
# Since this file is in func_test/, go up one level to get project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"[INFO] Loaded environment variables from {env_path}\n")
    else:
        print(f"[WARNING] .env file not found at {env_path}\n")
except ImportError:
    print("[WARNING] python-dotenv not installed, skipping .env file loading\n")

from actions.serpapi_search import SerpAPISearch
from actions.gcp_vertex_search import GCPVertexSearch
from config_manager import get_config


# ============================================================================
# TEST CONFIGURATION - Modify these parameters directly
# ============================================================================

# Search query
QUERY = "birthday of PRof. YIke GUO"

# Which providers to test: 'serpapi', 'gcp', or 'both'
PROVIDER = "both"

# Number of results to fetch (per provider)
# Note: Google Search API (SerpAPI) is limited to 10 results per request by Google's API
# To get more results, you would need pagination (multiple API calls)
NUM_RESULTS_SERPAPI = 10   # Google API hard limit
NUM_RESULTS_GCP = 10

# Number of results to display (None = show all)
DISPLAY_LIMIT = None

# Search engine (for SerpAPI)
ENGINE = "google"

# Language and location settings
LANGUAGE = "zh-cn"           # e.g., 'zh-cn', 'en', 'zh-tw'
LOCATION = "Hong Kong"       # e.g., 'Hong Kong', 'United States'
GL = "hk"                    # Country code: 'hk', 'us', 'cn', etc.

# Time filter (None or 'qdr:h', 'qdr:d', 'qdr:w', 'qdr:m', 'qdr:y')
# h=hour, d=day, w=week, m=month, y=year
# Note: SerpAPI filters results by time range
#       GCP sorts by date (orderBy=date) but returns all results
TIME_FILTER = None

# Target URLs to check for in results (optional)
TARGET_URLS = [
    # "hkust.edu.hk",
    # "sites.google.com",
]

# ============================================================================


def check_environment():
    """Check and display environment setup status"""
    print("=" * 80)
    print("ENVIRONMENT SETUP CHECK")
    print("=" * 80)
    
    # Load config
    cfg = get_config()
    
    # Check search providers from config
    providers = cfg.get('ir_rag.search.provider', 'serpapi')
    print(f"\nConfigured search providers (from config.yaml): {providers}")
    
    # Check SerpAPI
    serpapi_key = os.getenv('SERPAPI_KEY')
    print(f"\n[SerpAPI]")
    print(f"  Status: {'✅ Configured' if serpapi_key else '❌ Not configured'}")
    if not serpapi_key:
        print(f"  Setup: export SERPAPI_KEY='your_key_here'")
        print(f"  Get key: https://serpapi.com/")
    
    # Check GCP Vertex Search
    gcp_project = os.getenv('GCP_PROJECT_ID')
    gcp_engine = os.getenv('GCP_ENGINE_ID')
    gcp_key = os.getenv('GCP_API_KEY')
    
    print(f"\n[GCP Vertex Search]")
    print(f"  GCP_PROJECT_ID: {'✅ Set' if gcp_project else '❌ Not set'}")
    print(f"  GCP_ENGINE_ID: {'✅ Set' if gcp_engine else '❌ Not set'}")
    print(f"  GCP_API_KEY: {'✅ Set' if gcp_key else '❌ Not set'}")
    
    if not all([gcp_project, gcp_engine, gcp_key]):
        print(f"\n  Setup commands:")
        if not gcp_project:
            print(f"    export GCP_PROJECT_ID='your-project-id'")
        if not gcp_engine:
            print(f"    export GCP_ENGINE_ID='your-engine-id'")
        if not gcp_key:
            print(f"    export GCP_API_KEY='your-api-key'")
    
    # Check if any provider is configured
    has_serpapi = bool(serpapi_key)
    has_gcp = all([gcp_project, gcp_engine, gcp_key])
    
    print(f"\n{'='*80}")
    if not (has_serpapi or has_gcp):
        print("⚠️  WARNING: No search providers configured!")
        print("Please set up at least one search provider to run tests.")
        return False
    else:
        print(f"✅ Ready to test with: {', '.join([p for p, ok in [('SerpAPI', has_serpapi), ('GCP Vertex Search', has_gcp)] if ok])}")
        return True


def test_serpapi():
    """Test SerpAPI search"""
    print("=" * 80)
    print(f"[SerpAPI] Testing query: {QUERY}")
    print(f"  Parameters: engine={ENGINE}, language={LANGUAGE}, location={LOCATION}, gl={GL}")
    print(f"  time_filter={TIME_FILTER}, num_results={NUM_RESULTS_SERPAPI}")
    print("=" * 80)
    
    # Check if API key is configured
    if not os.getenv('SERPAPI_KEY'):
        print("\n[SerpAPI] ❌ SKIPPED - SERPAPI_KEY environment variable not set")
        print("   Set it with: export SERPAPI_KEY='your_key_here'")
        print("   Get your key at: https://serpapi.com/\n")
        return []
    
    searcher = SerpAPISearch()
    # get_structured_results returns list of dicts
    print(f"[DEBUG] Requesting {NUM_RESULTS_SERPAPI} results from SerpAPI...")
    results = searcher.get_structured_results(
        QUERY, 
        num_results=NUM_RESULTS_SERPAPI,
        engine=ENGINE,
        language=LANGUAGE,
        location=LOCATION,
        gl=GL,
        tbs=TIME_FILTER  # Use 'tbs' parameter, not 'time_filter'
    )
    
    print(f"\n[SerpAPI] Returned {len(results)} results")
    print(f"[NOTE] If SerpAPI returned only 10 results, this is a SerpAPI API limitation.")
    print(f"       The 'num' parameter in Google Search API is often limited to 10 results per request.")
    print(f"       To get more results, you may need to use pagination or upgrade your SerpAPI plan.\n")
    
    display = DISPLAY_LIMIT if DISPLAY_LIMIT else len(results)
    for i, result in enumerate(results[:display], 1):
        print(f"\n{i}. {result.get('title', 'No title')}")
        print(f"   URL: {result.get('link', 'No URL')}")
        print(f"   Source: {result.get('source', 'No source')}")
        print(f"   Type: {result.get('type', 'organic')}")
        print(f"   Position: {result.get('position', 0)}")
        if result.get('snippet'):
            snippet = result['snippet'][:150] + "..." if len(result['snippet']) > 150 else result['snippet']
            print(f"   Snippet: {snippet}")
    
    if display < len(results):
        print(f"\n... and {len(results) - display} more results (modify DISPLAY_LIMIT to show more)")
    
    return results


def test_gcp_vertex():
    """Test GCP Vertex Search"""
    print("\n\n" + "=" * 80)
    print(f"[GCP Vertex Search] Testing query: {QUERY}")
    print(f"  Parameters: language={LANGUAGE}, location={LOCATION}, num_results={NUM_RESULTS_GCP}")
    print(f"  time_filter={TIME_FILTER}")
    print("=" * 80)
    
    # Check if GCP is configured
    required_vars = ['GCP_PROJECT_ID', 'GCP_ENGINE_ID', 'GCP_API_KEY']
    missing = [v for v in required_vars if not os.getenv(v)]
    
    if missing:
        print(f"\n[GCP Vertex Search] ❌ SKIPPED - Missing: {', '.join(missing)}")
        print("   Set them with:")
        for var in missing:
            print(f"   export {var}='your_value_here'")
        print()
        return []
    
    searcher = GCPVertexSearch()
    # get_structured_results returns list of dicts
    results = searcher.get_structured_results(
        QUERY, 
        num_results=NUM_RESULTS_GCP,
        tbs=TIME_FILTER  # Pass time filter to GCP
    )
    
    print(f"\n[GCP Vertex Search] Returned {len(results)} results")
    if TIME_FILTER:
        print(f"\n[NOTE] GCP Vertex Search uses 'orderBy=date' for time-based sorting")
        print(f"       Results are sorted by date (most recent first) but not filtered by time range")
        print(f"       SerpAPI filters by time range: {TIME_FILTER}\n")
    else:
        print()
    
    display = DISPLAY_LIMIT if DISPLAY_LIMIT else len(results)
    for i, result in enumerate(results[:display], 1):
        print(f"\n{i}. {result.get('title', 'No title')}")
        print(f"   URL: {result.get('link', 'No URL')}")
        print(f"   Source: {result.get('source', 'No source')}")
        print(f"   Type: {result.get('type', 'organic')}")
        print(f"   Position: {result.get('position', 0)}")
        if result.get('snippet'):
            snippet = result['snippet'][:150] + "..." if len(result['snippet']) > 150 else result['snippet']
            print(f"   Snippet: {snippet}")
    
    if display < len(results):
        print(f"\n... and {len(results) - display} more results (modify DISPLAY_LIMIT to show more)")
    
    return results


def compare_results(serpapi_results, gcp_results):
    """Compare results from both search engines"""
    print("\n\n" + "=" * 80)
    print("[COMPARISON] Analyzing results")
    print("=" * 80)
    
    serpapi_urls = {r.get('link', '') for r in serpapi_results}
    gcp_urls = {r.get('link', '') for r in gcp_results}
    
    print(f"\nSerpAPI unique URLs: {len(serpapi_urls)}")
    print(f"GCP unique URLs: {len(gcp_urls)}")
    print(f"Common URLs: {len(serpapi_urls & gcp_urls)}")
    print(f"Only in SerpAPI: {len(serpapi_urls - gcp_urls)}")
    print(f"Only in GCP: {len(gcp_urls - serpapi_urls)}")


def main():
    """Main test function"""
    # Check environment setup first
    if not check_environment():
        print("\n❌ Cannot proceed without at least one search provider configured.")
        print("\nTo set up environment variables, use:")
        print("  export SERPAPI_KEY='your_key'")
        print("  # OR")
        print("  export GCP_PROJECT_ID='your_project'")
        print("  export GCP_ENGINE_ID='your_engine'")
        print("  export GCP_API_KEY='your_key'")
        return
    
    print("\n\nStarting tests...\n")
    
    # Display test configuration
    print("=" * 80)
    print("TEST CONFIGURATION")
    print("=" * 80)
    print(f"Query: {QUERY}")
    print(f"Provider: {PROVIDER}")
    print(f"Number of results - SerpAPI: {NUM_RESULTS_SERPAPI}, GCP: {NUM_RESULTS_GCP}")
    print(f"Display limit: {DISPLAY_LIMIT or 'all'}")
    print(f"Engine: {ENGINE}")
    print(f"Language: {LANGUAGE}")
    print(f"Location: {LOCATION}")
    print(f"GL: {GL}")
    print(f"Time filter: {TIME_FILTER or 'none'}")
    if TARGET_URLS:
        print(f"Target URLs: {', '.join(TARGET_URLS)}")
    print("=" * 80)
    
    print("\n\n" + "█" * 80)
    print(f"TESTING QUERY: {QUERY}")
    print("█" * 80)
    
    # Test search engines based on provider selection
    serpapi_results = []
    gcp_results = []
    
    if PROVIDER in ['serpapi', 'both']:
        serpapi_results = test_serpapi()
    
    if PROVIDER in ['gcp', 'both']:
        gcp_results = test_gcp_vertex()
    
    # Compare results if both providers were tested
    if PROVIDER == 'both' and (serpapi_results or gcp_results):
        compare_results(serpapi_results, gcp_results)
    
    # Check for target URLs if specified
    if TARGET_URLS:
        print("\n\n" + "=" * 80)
        print("[TARGET URLS CHECK]")
        print("=" * 80)
        for target in TARGET_URLS:
            found_serpapi = any(target in r.get('link', '') for r in serpapi_results)
            found_gcp = any(target in r.get('link', '') for r in gcp_results)
            
            print(f"\n{target}:")
            if PROVIDER in ['serpapi', 'both']:
                print(f"  Found in SerpAPI: {found_serpapi}")
                if found_serpapi:
                    match = next(r for r in serpapi_results if target in r.get('link', ''))
                    print(f"    Position: {match.get('position', 'N/A')}")
                    print(f"    Title: {match.get('title', 'N/A')}")
            
            if PROVIDER in ['gcp', 'both']:
                print(f"  Found in GCP: {found_gcp}")
                if found_gcp:
                    match = next(r for r in gcp_results if target in r.get('link', ''))
                    print(f"    Position: {match.get('position', 'N/A')}")
                    print(f"    Title: {match.get('title', 'N/A')}")
    
    print("\n" + "▬" * 80)
    print("\nTest completed.\n")


if __name__ == "__main__":
    main()


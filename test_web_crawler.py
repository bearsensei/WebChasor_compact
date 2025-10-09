#!/usr/bin/env python3
"""
Test script for WebVisitor (web crawler/scraper)
"""
import os
import sys
import asyncio

# Add project root and src to path
project_root = os.path.dirname(os.path.abspath(__file__))
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

from actions.visitor import WebVisitor, WebPage
from actions.ir_rag import IRConfig
from config_manager import get_config


# ============================================================================
# TEST CONFIGURATION - Modify these parameters directly
# ============================================================================

# Test mode: 'single' or 'batch'
TEST_MODE = "single"

# Single URL test
# Note: WebVisitor is designed for HTML pages. PDF/binary files may not work properly.
# Recommended: Use HTML pages like:
#   - "https://www.ird.gov.hk/chi/tax/sal.htm"
#   - "https://www.gov.hk/tc/residents/taxes/etax/services/tax_computation.htm"
SINGLE_URL = "https://hk.finance.yahoo.com/quote/NVDA/"

# Batch URLs test
BATCH_URLS = [
    "https://www.gov.hk/tc/residents/taxes/etax/services/tax_computation.htm",
    "https://www.gov.hk/tc/residents/taxes/taxfiling/taxrates/salariesrates.htm",
    "https://zh.wikipedia.org/wiki/香港稅務",
]

# Crawler configuration
FETCH_TIMEOUT = 10          # Seconds to wait for each page
MAX_CONCURRENT = 3          # Maximum concurrent requests
CHUNK_SIZE = 500            # Size of content chunks
CHUNK_OVERLAP = 50          # Overlap between chunks
MIN_PASSAGE_LENGTH = 20     # Minimum length of passages

# Display configuration
SHOW_FULL_CONTENT = True    # Show full content or just summary
MAX_CONTENT_DISPLAY = 5000  # Maximum characters to display per page

# Note: Wikipedia optimization and PDF extraction are now built-in to WebVisitor
# - Wikipedia pages automatically optimized (TOC, navbox, references removed)
# - PDF files automatically extracted (using PyMuPDF if available)
# - No configuration needed - automatic detection by URL/content-type

# ============================================================================


def print_section(title: str):
    """Print a section divider"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)




def print_page_info(page: WebPage, show_content: bool = False, max_chars: int = 500):
    """Print information about a fetched page"""
    status = "✅ SUCCESS" if page.success else "❌ FAILED"
    
    print(f"\n{status}")
    print(f"URL: {page.url}")
    print(f"Title: {page.title or 'No title'}")
    print(f"Status Code: {page.metadata.get('status_code', 'N/A')}")
    print(f"Content Type: {page.metadata.get('content_type', 'N/A')}")
    
    # Show extraction type if available
    extraction_type = page.metadata.get('extraction_type', 'standard')
    if extraction_type != 'standard':
        print(f"Extraction Type: {extraction_type}")
    
    # Show PDF-specific info
    if extraction_type == 'pdf':
        pages = page.metadata.get('pages', 'N/A')
        print(f"PDF Pages: {pages}")
    
    # Show Wikipedia-specific info
    if extraction_type == 'wikipedia_optimized':
        sections = page.metadata.get('sections_count', len(page.headings))
        print(f"Sections: {sections}")
        print(f"💡 Auto-optimized: TOC, infobox, navbox, references removed")
    
    print(f"Content Length: {len(page.content)} characters")
    print(f"Headings: {len(page.headings)} found")
    print(f"Fetch Time: {page.fetch_time:.2f} seconds")
    
    # Display headings
    if page.headings:
        print(f"\nHeadings/Sections (first 10):")
        for i, heading in enumerate(page.headings[:10], 1):
            heading_text = heading[:100] if isinstance(heading, str) else str(heading)[:100]
            print(f"  {i}. {heading_text}")
        if len(page.headings) > 10:
            print(f"  ... and {len(page.headings) - 10} more")
    
    # Display content
    if show_content and page.content:
        print(f"\nExtracted Content (first {max_chars} chars):")
        print("-" * 80)
        content_preview = page.content[:max_chars]
        if len(page.content) > max_chars:
            content_preview += f"\n\n... [showing {max_chars}/{len(page.content)} chars] ..."
        print(content_preview)
        print("-" * 80)
    
    # Display error if any
    if page.error:
        print(f"\nError: {page.error}")
    elif page.metadata.get('error'):
        print(f"\nError: {page.metadata['error']}")


async def test_single_url():
    """Test crawling a single URL"""
    print_section(f"TESTING SINGLE URL: {SINGLE_URL}")
    
    # Create configuration
    config = IRConfig(
        fetch_timeout=FETCH_TIMEOUT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_passage_length=MIN_PASSAGE_LENGTH
    )
    
    # Create visitor
    visitor = WebVisitor(config)
    
    print(f"\nFetching page...")
    print(f"  Timeout: {FETCH_TIMEOUT}s")
    print(f"  Chunk Size: {CHUNK_SIZE}")
    print(f"  Chunk Overlap: {CHUNK_OVERLAP}")
    
    # Fetch page (WebVisitor only has fetch_many, so use it with a single URL)
    pages = await visitor.fetch_many([SINGLE_URL])
    page = pages[0] if pages else None
    
    if not page:
        print("\n❌ Failed to fetch page")
        return None
    
    # Display results
    print_page_info(page, show_content=SHOW_FULL_CONTENT, max_chars=MAX_CONTENT_DISPLAY)
    
    return page


async def test_batch_urls():
    """Test crawling multiple URLs"""
    print_section(f"TESTING BATCH URLS: {len(BATCH_URLS)} URLs")
    
    # Create configuration
    config = IRConfig(
        fetch_timeout=FETCH_TIMEOUT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_passage_length=MIN_PASSAGE_LENGTH
    )
    
    # Create visitor
    visitor = WebVisitor(config)
    
    print(f"\nFetching {len(BATCH_URLS)} pages...")
    print(f"  Max Concurrent: {MAX_CONCURRENT}")
    print(f"  Timeout: {FETCH_TIMEOUT}s per page")
    print(f"  Chunk Size: {CHUNK_SIZE}")
    
    print(f"\nURLs to fetch:")
    for i, url in enumerate(BATCH_URLS, 1):
        print(f"  {i}. {url}")
    
    # Fetch pages
    pages = await visitor.fetch_many(BATCH_URLS)
    
    # Display results
    print_section("BATCH RESULTS")
    
    print(f"\nSummary:")
    print(f"  Total URLs: {len(BATCH_URLS)}")
    print(f"  Successfully fetched: {sum(1 for p in pages if p.success)}")
    print(f"  Failed: {sum(1 for p in pages if not p.success)}")
    
    # Display each page
    for i, page in enumerate(pages, 1):
        print(f"\n{'─' * 80}")
        print(f"PAGE {i}/{len(pages)}")
        print_page_info(page, show_content=SHOW_FULL_CONTENT, max_chars=MAX_CONTENT_DISPLAY)
    
    return pages


def print_statistics(pages):
    """Print statistics about crawled pages"""
    if not pages:
        return
    
    print_section("STATISTICS")
    
    successful_pages = [p for p in pages if p.success]
    failed_pages = [p for p in pages if not p.success]
    
    print(f"\nTotal Pages: {len(pages)}")
    print(f"  ✅ Successful: {len(successful_pages)}")
    print(f"  ❌ Failed: {len(failed_pages)}")
    
    if successful_pages:
        avg_size = sum(len(p.content) for p in successful_pages) / len(successful_pages)
        avg_time = sum(p.fetch_time for p in successful_pages) / len(successful_pages)
        total_headings = sum(len(p.headings) for p in successful_pages)
        
        print(f"\nSuccessful Pages Statistics:")
        print(f"  Average Content Size: {avg_size:.0f} characters")
        print(f"  Average Fetch Time: {avg_time:.2f} seconds")
        print(f"  Total Headings Found: {total_headings}")
        print(f"  Average Headings per Page: {total_headings / len(successful_pages):.1f}")
    
    if failed_pages:
        print(f"\nFailed Pages:")
        for i, page in enumerate(failed_pages, 1):
            error = page.metadata.get('error', 'Unknown error')
            print(f"  {i}. {page.url}")
            print(f"     Error: {error}")


async def main():
    """Main test function"""
    print_section("WEB CRAWLER TEST")
    
    print("\nConfiguration:")
    print(f"  Test Mode: {TEST_MODE}")
    print(f"  Fetch Timeout: {FETCH_TIMEOUT}s")
    print(f"  Chunk Size: {CHUNK_SIZE}")
    print(f"  Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"  Min Passage Length: {MIN_PASSAGE_LENGTH}")
    print(f"  Show Full Content: {SHOW_FULL_CONTENT}")
    print(f"\n💡 Auto Features:")
    print(f"  - Wikipedia optimization: ENABLED (built-in)")
    print(f"  - PDF text extraction: {'ENABLED' if 'fitz' in dir() else 'DISABLED (install pymupdf)'}")
    
    try:
        if TEST_MODE == "single":
            page = await test_single_url()
            pages = [page] if page else []
        elif TEST_MODE == "batch":
            pages = await test_batch_urls()
        else:
            print(f"\n❌ Invalid TEST_MODE: {TEST_MODE}")
            print("   Use 'single' or 'batch'")
            return
        
        # Print statistics
        if isinstance(pages, list) and len(pages) > 1:
            print_statistics(pages)
        
        print_section("TEST COMPLETED")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())


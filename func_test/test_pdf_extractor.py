#!/usr/bin/env python3
"""
Test script for PDF text extraction using PyMuPDF
"""
import os
import sys
import requests
import tempfile
from io import BytesIO

# Add project root and src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

try:
    import fitz  # PyMuPDF
    print(f"[INFO] PyMuPDF version: {fitz.version}\n")
except ImportError:
    print("[ERROR] PyMuPDF not installed. Install with: pip install pymupdf")
    sys.exit(1)


# ============================================================================
# TEST CONFIGURATION - Modify these parameters directly
# ============================================================================

# Test mode: 'single' or 'batch'
TEST_MODE = "single"

# Single PDF test
SINGLE_PDF_URL = "https://www.ird.gov.hk/chs/pdf/pam61c.pdf"

# Batch PDFs test
BATCH_PDF_URLS = [
    "https://www.ird.gov.hk/chs/pdf/pam61c.pdf",
    # Add more PDF URLs here
]

# Extraction configuration
EXTRACT_IMAGES = False          # Extract images from PDF
EXTRACT_METADATA = True         # Extract PDF metadata
SHOW_PAGE_BY_PAGE = False       # Show text page by page or as full document

# Display configuration
SHOW_FULL_TEXT = True           # Show full text or just preview
MAX_TEXT_DISPLAY = 5000         # Maximum characters to display
SAVE_TO_FILE = False            # Save extracted text to file
OUTPUT_DIR = "output"           # Directory to save extracted text

# Download configuration
DOWNLOAD_TIMEOUT = 30           # Seconds to wait for PDF download
TEMP_DIR = None                 # Use system temp dir (None) or specify path

# ============================================================================


def print_section(title: str):
    """Print a section divider"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def download_pdf(url: str, timeout: int = 30) -> bytes:
    """Download PDF from URL"""
    print(f"📥 Downloading PDF from: {url}")
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower():
            print(f"⚠️  Warning: Content-Type is '{content_type}', expected PDF")
        
        print(f"✅ Downloaded {len(response.content)} bytes")
        return response.content
    except requests.RequestException as e:
        print(f"❌ Download failed: {e}")
        return None


def extract_pdf_metadata(doc) -> dict:
    """Extract metadata from PDF document"""
    metadata = {
        'title': doc.metadata.get('title', 'N/A'),
        'author': doc.metadata.get('author', 'N/A'),
        'subject': doc.metadata.get('subject', 'N/A'),
        'keywords': doc.metadata.get('keywords', 'N/A'),
        'creator': doc.metadata.get('creator', 'N/A'),
        'producer': doc.metadata.get('producer', 'N/A'),
        'creation_date': doc.metadata.get('creationDate', 'N/A'),
        'modification_date': doc.metadata.get('modDate', 'N/A'),
        'pages': doc.page_count,
        'encrypted': doc.is_encrypted,
    }
    return metadata


def extract_text_from_pdf(pdf_bytes: bytes, page_by_page: bool = False) -> dict:
    """
    Extract text from PDF bytes using PyMuPDF
    
    Returns:
        dict with 'text', 'pages', 'metadata', etc.
    """
    try:
        # Open PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        result = {
            'success': True,
            'pages': doc.page_count,
            'metadata': extract_pdf_metadata(doc),
            'text': '',
            'page_texts': [],
            'total_chars': 0,
            'error': None
        }
        
        # Extract text from each page
        all_text = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            result['page_texts'].append({
                'page': page_num + 1,
                'text': page_text,
                'chars': len(page_text)
            })
            all_text.append(page_text)
        
        # Combine all text
        result['text'] = '\n'.join(all_text)
        result['total_chars'] = len(result['text'])
        
        doc.close()
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'pages': 0,
            'text': '',
            'metadata': {},
            'page_texts': [],
            'total_chars': 0
        }


def print_pdf_info(url: str, result: dict, show_text: bool = True, max_chars: int = 5000):
    """Print information about extracted PDF"""
    if not result['success']:
        print(f"\n❌ FAILED: {url}")
        print(f"Error: {result['error']}")
        return
    
    print(f"\n✅ SUCCESS: {url}")
    print(f"\nPages: {result['pages']}")
    print(f"Total Characters: {result['total_chars']:,}")
    
    # Print metadata
    if result['metadata']:
        print(f"\nMetadata:")
        for key, value in result['metadata'].items():
            if value and value != 'N/A':
                print(f"  {key.title()}: {value}")
    
    # Print page statistics
    if result['page_texts']:
        print(f"\nPage Statistics:")
        for page_info in result['page_texts'][:5]:  # Show first 5 pages
            print(f"  Page {page_info['page']}: {page_info['chars']} characters")
        if len(result['page_texts']) > 5:
            print(f"  ... and {len(result['page_texts']) - 5} more pages")
    
    # Print text content
    if show_text and result['text']:
        print(f"\nExtracted Text (first {max_chars} chars):")
        print("-" * 80)
        text_preview = result['text'][:max_chars]
        if len(result['text']) > max_chars:
            text_preview += "\n\n... [truncated] ..."
        print(text_preview)
        print("-" * 80)


def save_text_to_file(url: str, text: str, output_dir: str = "output") -> str:
    """Save extracted text to file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename from URL
    filename = url.split('/')[-1].replace('.pdf', '.txt')
    filepath = os.path.join(output_dir, filename)
    
    # Save text
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Source: {url}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(text)
    
    print(f"💾 Saved to: {filepath}")
    return filepath


def test_single_pdf(url: str):
    """Test extracting text from a single PDF"""
    print_section(f"TESTING SINGLE PDF")
    print(f"URL: {url}")
    
    # Download PDF
    pdf_bytes = download_pdf(url, timeout=DOWNLOAD_TIMEOUT)
    if not pdf_bytes:
        return None
    
    # Extract text
    print(f"\n📄 Extracting text from PDF...")
    result = extract_text_from_pdf(pdf_bytes, page_by_page=SHOW_PAGE_BY_PAGE)
    
    # Print results
    print_pdf_info(url, result, show_text=SHOW_FULL_TEXT, max_chars=MAX_TEXT_DISPLAY)
    
    # Save to file if requested
    if SAVE_TO_FILE and result['success'] and result['text']:
        save_text_to_file(url, result['text'], OUTPUT_DIR)
    
    return result


def test_batch_pdfs(urls: list):
    """Test extracting text from multiple PDFs"""
    print_section(f"TESTING BATCH PDFs: {len(urls)} PDFs")
    
    results = []
    for i, url in enumerate(urls, 1):
        print(f"\n{'─' * 80}")
        print(f"PDF {i}/{len(urls)}")
        
        # Download PDF
        pdf_bytes = download_pdf(url, timeout=DOWNLOAD_TIMEOUT)
        if not pdf_bytes:
            results.append({'success': False, 'url': url, 'error': 'Download failed'})
            continue
        
        # Extract text
        print(f"📄 Extracting text from PDF...")
        result = extract_text_from_pdf(pdf_bytes, page_by_page=SHOW_PAGE_BY_PAGE)
        result['url'] = url
        results.append(result)
        
        # Print results
        print_pdf_info(url, result, show_text=SHOW_FULL_TEXT, max_chars=MAX_TEXT_DISPLAY)
        
        # Save to file if requested
        if SAVE_TO_FILE and result['success'] and result['text']:
            save_text_to_file(url, result['text'], OUTPUT_DIR)
    
    return results


def print_statistics(results: list):
    """Print statistics about extracted PDFs"""
    print_section("STATISTICS")
    
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    print(f"\nTotal PDFs: {len(results)}")
    print(f"  ✅ Successful: {len(successful)}")
    print(f"  ❌ Failed: {len(failed)}")
    
    if successful:
        total_pages = sum(r.get('pages', 0) for r in successful)
        total_chars = sum(r.get('total_chars', 0) for r in successful)
        avg_pages = total_pages / len(successful)
        avg_chars = total_chars / len(successful)
        
        print(f"\nSuccessful PDFs Statistics:")
        print(f"  Total Pages: {total_pages}")
        print(f"  Total Characters: {total_chars:,}")
        print(f"  Average Pages per PDF: {avg_pages:.1f}")
        print(f"  Average Characters per PDF: {avg_chars:,.0f}")
    
    if failed:
        print(f"\nFailed PDFs:")
        for i, result in enumerate(failed, 1):
            print(f"  {i}. {result.get('url', 'Unknown')}")
            print(f"     Error: {result.get('error', 'Unknown error')}")


def main():
    """Main test function"""
    print_section("PDF TEXT EXTRACTION TEST")
    
    print("\nConfiguration:")
    print(f"  Test Mode: {TEST_MODE}")
    print(f"  Download Timeout: {DOWNLOAD_TIMEOUT}s")
    print(f"  Extract Metadata: {EXTRACT_METADATA}")
    print(f"  Show Page by Page: {SHOW_PAGE_BY_PAGE}")
    print(f"  Show Full Text: {SHOW_FULL_TEXT}")
    print(f"  Max Text Display: {MAX_TEXT_DISPLAY} chars")
    print(f"  Save to File: {SAVE_TO_FILE}")
    if SAVE_TO_FILE:
        print(f"  Output Directory: {OUTPUT_DIR}")
    
    try:
        if TEST_MODE == "single":
            result = test_single_pdf(SINGLE_PDF_URL)
            results = [result] if result else []
        elif TEST_MODE == "batch":
            results = test_batch_pdfs(BATCH_PDF_URLS)
        else:
            print(f"\n❌ Invalid TEST_MODE: {TEST_MODE}")
            print("   Use 'single' or 'batch'")
            return
        
        # Print statistics for batch mode
        if TEST_MODE == "batch" and len(results) > 1:
            print_statistics(results)
        
        print_section("TEST COMPLETED")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


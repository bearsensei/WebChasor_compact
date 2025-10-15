"""
Web Visitor Module
Fetches and cleans web page content for information retrieval.
Supports: HTML pages, Wikipedia optimization, PDF text extraction
"""

import re
import time
import asyncio
import logging
import requests
from typing import List
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

# Optional PDF support
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.info("PyMuPDF not available. PDF extraction disabled.")


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class WebPage:
    """Fetched and cleaned web page content"""
    url: str
    title: str
    content: str
    headings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fetch_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


# ============================================================================
# WebVisitor Class
# ============================================================================

class WebVisitor:
    """Component for fetching and cleaning web pages"""
    
    def __init__(self, config):
        """
        Initialize WebVisitor with configuration.
        
        Args:
            config: IRConfig object with fetch_timeout and max_pages_to_visit
        """
        self.config = config
    
    async def fetch_many(self, urls: List[str]) -> List[WebPage]:
        """Fetch multiple URLs in parallel"""
        print(f"🌐 VISITOR: Fetching {len(urls)} pages...")
        
        # Limit to max pages
        urls = urls[:self.config.max_pages_to_visit]
        
        # Create tasks for parallel fetching
        tasks = [self._fetch_single(url) for url in urls]
        pages = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful pages
        successful_pages = []
        for page in pages:
            if isinstance(page, WebPage) and page.success:
                successful_pages.append(page)
                print(f"🌐 VISITOR: Successfully fetched {page.url}")
        
        print(f"🌐 VISITOR: Successfully fetched {len(successful_pages)}/{len(urls)} pages")
        return successful_pages
    
    async def _fetch_single(self, url: str) -> WebPage:
        """
        Fetch and clean a single web page
        Automatically handles: PDF files, Wikipedia pages, and standard HTML
        """
        start_time = time.time()
        
        try:
            # Basic URL validation
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return WebPage(
                    url=url, title="", content="", success=False, 
                    error="Invalid URL", fetch_time=time.time() - start_time
                )
            
            # Fetch page content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=self.config.fetch_timeout)
            response.raise_for_status()
            
            # Check content type and route to appropriate handler
            content_type = response.headers.get('content-type', '').lower()
            
            # PDF handling
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                logger.info(f"[VISITOR][PDF] Detected PDF: {url}")
                return self._extract_pdf(url, response.content, start_time)
            
            # HTML handling
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if it's Wikipedia and apply optimization
            if 'wikipedia.org' in url.lower():
                logger.info(f"[VISITOR][WIKI] Detected Wikipedia page: {url}")
                return self._extract_wikipedia(url, soup, response, start_time)
            
            # Standard HTML extraction
            return self._extract_html(url, soup, response, start_time)
            
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return WebPage(
                url=url, title="", content="", success=False,
                error=str(e), fetch_time=time.time() - start_time
            )
    
    def _extract_html(self, url: str, soup: BeautifulSoup, response, start_time: float) -> WebPage:
        """Extract content from standard HTML page"""
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ""
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article'))
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text content
        content = main_content.get_text(separator='\n', strip=True)
        
        # Extract headings for context
        headings = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            heading_text = heading.get_text().strip()
            if heading_text:
                headings.append(heading_text)
        
        # Clean content
        content = self._clean_content(content)
        
        return WebPage(
            url=url,
            title=title,
            content=content,
            headings=headings,
            metadata={'status_code': response.status_code, 'content_type': response.headers.get('content-type', '')},
            fetch_time=time.time() - start_time,
            success=True
        )
    
    def _extract_wikipedia(self, url: str, soup: BeautifulSoup, response, start_time: float) -> WebPage:
        """
        Extract optimized content from Wikipedia pages
        Removes: TOC, infobox, navbox, references, external links
        Keeps: Main text, section headings, lists
        """
        # Extract title from Wikipedia's specific heading
        title_elem = soup.find('h1', {'id': 'firstHeading'})
        title = title_elem.get_text().strip() if title_elem else soup.find('title').get_text().strip()
        
        # Find Wikipedia main content area
        main_content = soup.find('div', {'class': 'mw-parser-output'})
        if not main_content:
            main_content = soup.find('div', {'id': 'mw-content-text'})
        
        if not main_content:
            logger.warning(f"[VISITOR][WIKI] Could not find Wikipedia content, using standard extraction")
            return self._extract_html(url, soup, response, start_time)
        
        # Remove Wikipedia-specific noise elements
        # NOTE: Keep table.wikitable (content tables), only remove navigation/info tables
        noise_selectors = [
            'table.infobox', 'table.navbox', 'div.navbox', 'table.vertical-navbox',
            'div.toc', 'div#toc', 'div.toccolours',
            'div.reflist', 'div.refbegin', 'ol.references',
            'div.catlinks', 'div.printfooter', 'div.mw-authority-control',
            'div.mw-editsection', 'span.mw-editsection',
            'div.hatnote', 'div.ambox',
            'div.side-box', 'div.metadata',
        ]
        
        for selector in noise_selectors:
            for elem in main_content.select(selector):
                elem.decompose()
        
        # Remove elements with navigation/menu classes
        for elem in main_content.find_all(class_=re.compile(r'(navbox|toc|sidebar|nav|menu)', re.I)):
            elem.decompose()
        
        # Remove "目录" section
        for elem in main_content.find_all(string=re.compile(r'^(目录|目錄|Contents)$')):
            parent = elem.find_parent()
            if parent:
                parent.decompose()
        
        # Remove end sections (references, external links, etc.) and everything after
        end_section_markers = ['参考资料', '參考資料', '外部链接', '外部連結', '参见', '參見', '注释', '註釋', '延伸阅读', '延伸閱讀']
        for h2 in main_content.find_all('h2'):
            heading_text = h2.get_text().strip()
            heading_text = re.sub(r'\[编辑\]|\[編輯\]', '', heading_text).strip()
            
            if heading_text in end_section_markers:
                # Remove this heading and all following siblings
                for sibling in list(h2.find_next_siblings()):
                    sibling.decompose()
                h2.decompose()
                break
        
        # Extract section headings
        headings = []
        for h in main_content.find_all(['h2', 'h3']):
            heading_text = h.get_text().strip()
            heading_text = re.sub(r'\[编辑\]|\[編輯\]', '', heading_text).strip()
            if heading_text:
                headings.append(heading_text)
        
        # Extract clean text from paragraphs, lists, and tables
        content_parts = []
        for elem in main_content.find_all(['p', 'h2', 'h3', 'ul', 'ol', 'table']):
            if elem.name in ['h2', 'h3']:
                heading_text = elem.get_text(strip=True)
                heading_text = re.sub(r'\[编辑\]|\[編輯\]', '', heading_text).strip()
                if heading_text:
                    content_parts.append(f"\n## {heading_text}\n")
            elif elem.name == 'table':
                # Extract table content (wikitable class indicates content table)
                if 'wikitable' in elem.get('class', []):
                    table_text = self._extract_table_text(elem)
                    if table_text and len(table_text) > 20:
                        content_parts.append(f"\n[TABLE]\n{table_text}\n[/TABLE]\n")
            else:
                text = elem.get_text(separator=' ', strip=True)
                if text and len(text) > 10:
                    content_parts.append(text)
        
        content = '\n\n'.join(content_parts)
        
        # Clean up citation markers and edit links
        content = re.sub(r'\[\d+\]', '', content)
        content = re.sub(r'\[编辑\]|\[編輯\]', '', content)
        content = re.sub(r'\n\s*\n+', '\n\n', content)
        content = content.strip()
        
        logger.info(f"[VISITOR][WIKI] Extracted {len(content)} chars from {len(headings)} sections")
        
        return WebPage(
            url=url,
            title=title,
            content=content,
            headings=headings,
            metadata={
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'extraction_type': 'wikipedia_optimized',
                'sections_count': len(headings)
            },
            fetch_time=time.time() - start_time,
            success=True
        )
    
    def _extract_pdf(self, url: str, pdf_bytes: bytes, start_time: float) -> WebPage:
        """
        Extract text from PDF using PyMuPDF
        """
        if not PDF_SUPPORT:
            logger.warning(f"[VISITOR][PDF] PyMuPDF not available, cannot extract PDF: {url}")
            return WebPage(
                url=url, title="", content="",
                metadata={'content_type': 'application/pdf'},
                success=False,
                error="PyMuPDF not installed. Install with: pip install pymupdf",
                fetch_time=time.time() - start_time
            )
        
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Extract metadata (before closing doc)
            title = doc.metadata.get('title', '') or url.split('/')[-1].replace('.pdf', '')
            page_count = doc.page_count
            pdf_metadata = dict(doc.metadata)
            
            # Extract text from all pages
            all_text = []
            for page_num in range(page_count):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text and page_text.strip():
                    all_text.append(page_text)
            
            content = '\n\n'.join(all_text)
            doc.close()
            
            # Extract simple headings (lines that look like headings)
            headings = []
            for line in content.split('\n'):
                line = line.strip()
                # Simple heuristic: short lines in title case might be headings
                if line and len(line) < 100 and len(line.split()) < 10:
                    if line and (line[0].isupper() if line else False):
                        headings.append(line)
                    elif any(c in line for c in '一二三四五六七八九十'):
                        headings.append(line)
            
            logger.info(f"[VISITOR][PDF] Extracted {len(content)} chars from {page_count} pages")
            
            return WebPage(
                url=url,
                title=title,
                content=content,
                headings=headings[:20],  # Limit headings from PDF heuristic
                metadata={
                    'content_type': 'application/pdf',
                    'extraction_type': 'pdf',
                    'pages': page_count,
                    'pdf_metadata': pdf_metadata
                },
                fetch_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"[VISITOR][PDF] Failed to extract PDF {url}: {e}")
            return WebPage(
                url=url, title="", content="",
                metadata={'content_type': 'application/pdf'},
                success=False,
                error=f"PDF extraction failed: {str(e)}",
                fetch_time=time.time() - start_time
            )
    
    def _extract_table_text(self, table_elem) -> str:
        """
        Extract text content from a table element and format it for readability.
        Converts HTML table to structured text format.
        
        Args:
            table_elem: BeautifulSoup table element
            
        Returns:
            Formatted table text
        """
        try:
            rows = []
            
            # Extract table caption if available
            caption = table_elem.find('caption')
            if caption:
                caption_text = caption.get_text(strip=True)
                if caption_text:
                    rows.append(f"表格标题: {caption_text}")
                    rows.append("")
            
            # Extract headers
            headers = []
            thead = table_elem.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    for th in header_row.find_all(['th', 'td']):
                        headers.append(th.get_text(strip=True))
            else:
                # Sometimes headers are in first row of tbody
                first_row = table_elem.find('tr')
                if first_row:
                    ths = first_row.find_all('th')
                    if ths:
                        for th in ths:
                            headers.append(th.get_text(strip=True))
            
            if headers:
                rows.append(" | ".join(headers))
                rows.append("-" * (len(" | ".join(headers))))
            
            # Extract data rows
            tbody = table_elem.find('tbody') or table_elem
            for tr in tbody.find_all('tr'):
                # Skip header rows in tbody
                if tr.find('th') and not tr.find('td'):
                    continue
                
                cells = []
                for cell in tr.find_all(['td', 'th']):
                    cell_text = cell.get_text(separator=' ', strip=True)
                    # Remove citation markers
                    cell_text = re.sub(r'\[\d+\]', '', cell_text)
                    cells.append(cell_text)
                
                if cells and any(c.strip() for c in cells):  # Skip empty rows
                    rows.append(" | ".join(cells))
            
            return "\n".join(rows)
            
        except Exception as e:
            logger.warning(f"[VISITOR][TABLE] Failed to extract table: {e}")
            # Fallback: just get all text
            return table_elem.get_text(separator=' | ', strip=True)
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        # Remove common boilerplate patterns
        patterns_to_remove = [
            r'Cookie Policy.*?Accept',
            r'Subscribe to.*?newsletter',
            r'Follow us on.*?social media',
            r'Advertisement\s*',
            r'Sponsored content.*?\n',
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        
        return content.strip()


# ============================================================================
# Test/Demo Code
# ============================================================================

if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        max_pages_to_visit: int = 3
        fetch_timeout: int = 10
    
    async def test_visitor():
        """Test WebVisitor with sample URLs"""
        config = MockConfig()
        visitor = WebVisitor(config)
        
        # Test URLs
        test_urls = [
            "https://zh.wikipedia.org/wiki/Python",
            "https://www.python.org/about/",
            "https://docs.python.org/3/"
        ]
        
        print("[TEST][Visitor] Testing web page fetching...")
        print("=" * 80)
        
        pages = await visitor.fetch_many(test_urls)
        
        print("\n" + "=" * 80)
        print(f"[TEST][Result] Fetched {len(pages)}/{len(test_urls)} pages successfully")
        print("\n[TEST][Pages]:")
        for i, page in enumerate(pages, 1):
            print(f"  {i}. {page.title}")
            print(f"     URL: {page.url}")
            print(f"     Content length: {len(page.content)} chars")
            print(f"     Headings: {len(page.headings)}")
            print(f"     Fetch time: {page.fetch_time:.2f}s")
        print("=" * 80)
    
    # Run test
    asyncio.run(test_visitor())


"""
Content Ranker Module
Ranks and chunks content passages by relevance to extraction tasks.
"""

import re
from typing import List, Dict
from dataclasses import dataclass, field
from typing import Any

# Import from planner module
from planner import PlanTask, ExtractionPlan

# Import from visitor module
from actions.visitor import WebPage


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ContentPassage:
    """Ranked content passage for extraction"""
    text: str
    # Original formatting-preserved text (keeps markdown/html tables, line breaks)
    raw_text: str = ""
    source_url: str = ""
    heading_context: str = ""
    score: float = 0.0
    task_relevance: Dict[str, float] = field(default_factory=dict)
    position: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# ContentRanker Class
# ============================================================================

class ContentRanker:
    """Component for ranking and chunking content passages"""
    
    def __init__(self, config):
        """
        Initialize ContentRanker with configuration.
        
        Args:
            config: IRConfig object with chunk_size, chunk_overlap, max_passages_per_task
        """
        self.config = config
    
    def rank_passages(self, plan: ExtractionPlan, pages: List[WebPage]) -> Dict[str, List[ContentPassage]]:
        """Rank content passages by relevance to each extraction task"""
        print(f"📊 RANKER: Ranking passages for {len(plan.tasks_to_extract)} tasks across {len(pages)} pages...")
        
        task_passages = {}
        
        for task in plan.tasks_to_extract:
            passages = []
            
            # Extract and rank passages for this task
            for page in pages:
                page_passages = self._extract_passages_from_page(page, task)
                passages.extend(page_passages)
            
            # Sort by relevance score
            passages.sort(key=lambda p: p.score, reverse=True)
            
            # Keep top passages
            task_passages[task.variable_name] = passages[:self.config.max_passages_per_task]
            
            print(f"📊 RANKER: Task '{task.variable_name}' has {len(task_passages[task.variable_name])} top passages")
        
        return task_passages
    
    def _extract_passages_from_page(self, page: WebPage, task: PlanTask) -> List[ContentPassage]:
        """Extract and score passages from a single page for a task"""
        passages = []
        
        # Split content into chunks (both sentence-based and formatting-preserved)
        chunks, preserved_chunks = self._chunk_content_pair(page.content)
        
        for i, chunk in enumerate(chunks):
            # Calculate relevance score
            score = self._calculate_relevance_score(chunk, task, page)
            
            # Find relevant heading context
            heading_context = self._find_heading_context(chunk, page.headings)

            page_meta = dict(page.metadata or {})
            page_meta['chunk_index'] = i
            page_meta.setdefault('page_title', page.title)
            page_meta.setdefault('original_url', page.url)
            if page_meta.get('pseudo_source') == 'SERP_SNIPPET':
                page_meta.setdefault('provenance', f"SERP_SNIPPET::{page_meta.get('original_url', page.url)}")
            else:
                page_meta.setdefault('provenance', page.url)

            source_url = page_meta.get('pseudo_source', page.url)
            
            passage = ContentPassage(
                text=chunk,
                raw_text=preserved_chunks[i],
                source_url=source_url,
                heading_context=heading_context,
                score=score,
                position=i,
                metadata=page_meta
            )
            
            passages.append(passage)

        return passages
    
    def _chunk_content_pair(self, content: str) -> (List[str], List[str]):
        """
        Produce two parallel chunk lists:
        - chunks: sentence-based (good for relevance scoring)
        - preserved_chunks: structure-preserving (keeps newlines/markdown tables)
        Both lists are aligned by index for downstream usage.
        """
        # -------- sentence-based chunks (current behavior) --------
        sentences = re.split(r'[.!?]+', content)
        chunks: List[str] = []
        current_chunk = ""
        current_length = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_length = len(sentence.split())
            if current_length + sentence_length > self.config.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # overlap with last 2 "sentences" (approx by periods)
                overlap_sentences = current_chunk.split('.')[-2:]
                current_chunk = '. '.join([s for s in overlap_sentences if s]) + ('. ' if overlap_sentences else '')
                current_chunk += sentence
                current_length = len(current_chunk.split())
            else:
                current_chunk += ('. ' if current_chunk else '') + sentence
                current_length += sentence_length
        if current_chunk:
            chunks.append(current_chunk.strip())

        # -------- structure-preserving chunks --------
        # Strategy: split by double newlines (paragraphs) so markdown/html tables and line breaks are retained.
        # We then accumulate paragraphs until we reach roughly the same token budget as sentence chunks.
        paras = [p for p in re.split(r'\n{2,}', content) if p.strip()]
        preserved_chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0
        target = max(1, self.config.chunk_size)  # word budget analogous to sentence chunks
        for p in paras:
            # Keep paragraph exactly (retain internal newlines)
            words = len(p.split())
            # if paragraph is a markdown table or contains '[TABLE]', we try not to split it
            is_table_like = ('| ' in p or '\n|' in p or '[TABLE]' in p or '&lt;table' in p or '&lt;tr' in p or '<table' in p)
            if buf and (buf_len + words > target) and not is_table_like:
                preserved_chunks.append('\n\n'.join(buf).strip())
                buf = [p]
                buf_len = words
            else:
                buf.append(p)
                buf_len += words

            # If the buffer is far over budget and current paragraph is not table-like, flush
            if buf_len >= target * 1.3 and not is_table_like:
                preserved_chunks.append('\n\n'.join(buf).strip())
                buf, buf_len = [], 0

        if buf:
            preserved_chunks.append('\n\n'.join(buf).strip())

        # Align lengths: if mismatch, fall back to a safe alignment by rebalancing
        # We pad the shorter list with last element to ensure same length (non-destructive).
        if not preserved_chunks:
            preserved_chunks = [content.strip()]
        if not chunks:
            chunks = [content.strip()]
        if len(preserved_chunks) != len(chunks):
            if len(preserved_chunks) < len(chunks):
                preserved_chunks += [preserved_chunks[-1]] * (len(chunks) - len(preserved_chunks))
            else:
                chunks += [chunks[-1]] * (len(preserved_chunks) - len(chunks))

        return chunks, preserved_chunks

    # Backwards-compatible wrapper used by older callers (if any)
    def _chunk_content(self, content: str) -> List[str]:
        chunks, _ = self._chunk_content_pair(content)
        return chunks
    
    def _calculate_relevance_score(self, chunk: str, task: PlanTask, page: WebPage) -> float:
        """Calculate relevance score for a chunk"""
        score = 0.0
        chunk_lower = chunk.lower()
        task_lower = task.fact.lower()
        
        # Extract keywords from task (both fact and variable_name) with multilingual support
        task_text = task_lower + ' ' + task.variable_name.replace('_', ' ').lower()
        
        # Multilingual keyword extraction: English words + Chinese characters
        english_keywords = re.findall(r'\b[a-z]{3,}\b', task_text)  # English words (3+ chars)
        chinese_keywords = re.findall(r'[\u4e00-\u9fff]+', task_text)  # Chinese characters
        
        all_keywords = set(english_keywords + chinese_keywords)
        
        # Base keyword matching with dynamic weighting
        keyword_base_weight = self._get_keyword_weight_for_category(task.category)
        for keyword in all_keywords:
            if keyword in chunk_lower:
                score += keyword_base_weight
        
        # Entity matching (if available) - universal across languages
        if hasattr(task, 'entity') and task.entity:
            entity_lower = task.entity.lower()
            if entity_lower in chunk_lower:
                entity_weight = self.config.entity_weight if hasattr(self.config, 'entity_weight') else 3.5
                score += entity_weight
        
        # Detect if task requires structured data (lists, rosters, tables)
        # Based on task content, not hardcoded keywords
        requires_structured = self._task_requires_structured_data(task)
        
        if requires_structured:
            # Boost for structured content patterns
            if '[TABLE]' in chunk:  # Table markers from visitor
                # Strong boost for tables in list/roster tasks
                table_count = chunk.count('[TABLE]')
                score += 5.0 * table_count  # Increased from 3.0
                
                # Extra boost if table has multiple rows (high-value structured data)
                pipe_count = chunk.count(' | ')
                if pipe_count > 10:  # Dense table
                    score += 3.0
                    
            if re.search(r'^\s*[-•*]\s', chunk, re.MULTILINE):  # Lists
                score += 2.0
                
            # Detect name patterns (Chinese 2-4 chars, or Western capitalized names)
            chinese_names = len(re.findall(r'[\u4e00-\u9fff]{2,4}(?=[：:])', chunk))
            western_names = len(re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', chunk))
            score += (chinese_names + western_names) * 0.5
        
        # Temporal information detection (if task needs dates/years)
        if self._task_needs_temporal(task):
            year_count = len(re.findall(r'\d{4}', chunk))
            date_count = len(re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', chunk))
            score += (year_count + date_count) * 0.3
        
        # Length adjustment (dynamic based on task)
        words = chunk.split()
        if len(words) < 20:
            score *= 0.7  # Penalize very short chunks
        elif len(words) > 100 and requires_structured:
            score *= 1.1  # Reward long structured content
        
        # Apply source-based weighting
        source_weight = self._get_source_weight(page, task)
        score *= source_weight
        
        return score
    
    def _get_keyword_weight_for_category(self, category: str) -> float:
        """Dynamically adjust keyword weight based on task category"""
        # Categories requiring higher precision
        high_precision_categories = {'fact_verification', 'comparison'}
        # Categories requiring comprehensive coverage
        comprehensive_categories = {'aggregation', 'biography', 'background'}
        
        if category in high_precision_categories:
            return 1.5  # Higher weight for precise matching
        elif category in comprehensive_categories:
            return 1.0  # Standard weight for broad coverage
        else:
            return 1.2  # Default
    
    def _task_requires_structured_data(self, task: PlanTask) -> bool:
        """
        Detect if task requires structured data (lists, tables, rosters)
        Based on semantic analysis, not hardcoded keywords
        """
        task_text = (task.fact + ' ' + task.variable_name).lower()
        
        # Detect list-related patterns (universal across languages)
        list_indicators = [
            'list', '列', '名单', 'roster', 'members',
            'all', '所有', 'each', '各', '每',
            'chiefs', 'heads', 'leaders', '司长', '局长', '负责人',
            'who are', '谁是', '有哪些', 'what are'
        ]
        
        return any(indicator in task_text for indicator in list_indicators)
    
    def _task_needs_temporal(self, task: PlanTask) -> bool:
        """Detect if task requires temporal information"""
        task_text = (task.fact + ' ' + task.variable_name).lower()
        
        # Temporal indicators
        temporal_indicators = [
            'when', 'year', 'date', 'time', '何时', '年', '时间',
            'appointed', 'started', 'founded', 'established',
            '任命', '就职', '成立', '建立', 'timeline', '时间线'
        ]
        
        return any(indicator in task_text for indicator in temporal_indicators)
    
    def _get_source_weight(self, page: WebPage, task: PlanTask = None) -> float:
        """
        Calculate source weight based on content type and quality indicators.
        Dynamically adjusts based on task requirements and page metadata.
        """
        metadata = page.metadata or {}
        
        # Base weight = 1.0
        weight = 1.0
        
        # Factor 1: Content type priority
        source_type = metadata.get('source_type', 'unknown')
        extraction_type = metadata.get('extraction_type', 'standard')
        
        if source_type == 'snippet':
            # Snippets are good for quick facts but lack context
            weight = 1.0
        elif source_type == 'webpage':
            # Full web pages have more context
            weight = 1.3  # Increased from 1.2
            
            # Factor 2: Content richness (based on extraction_type metadata)
            if extraction_type in ('wikipedia', 'wikipedia_optimized'):
                # Wikipedia: encyclopedic, structured, reliable
                base_wiki_weight = 2.5  # Increased from 2.0
                
                # Dynamic boost for tasks that benefit from encyclopedia content
                if task and task.category in ('background', 'definition', 'aggregation', 'biography'):
                    weight = base_wiki_weight * 1.2  # Extra boost for these categories
                else:
                    weight = base_wiki_weight
                    
            elif extraction_type == 'pdf':
                weight = 1.4  # PDF documents (official/academic)
        
        # Factor 3: Content quality indicators from metadata
        content_length = len(page.content.split())
        
        # Reward rich content (with diminishing returns)
        if content_length > 1000:
            weight *= 1.15
        elif content_length > 500:
            weight *= 1.1
        elif content_length < 100:
            weight *= 0.7
        
        # Factor 4: Structured content indicators
        if metadata.get('has_infobox'):
            weight *= 1.15  # Infoboxes contain dense factual information
        
        section_count = metadata.get('section_count', 0) or metadata.get('sections_count', 0)
        if section_count > 10:
            weight *= 1.15  # Highly structured
        elif section_count > 5:
            weight *= 1.1
        
        # Factor 5: Table content (if present)
        if '[TABLE]' in page.content:
            # Tables often contain the structured data we need
            table_count = page.content.count('[TABLE]')
            weight *= (1.0 + min(table_count * 0.1, 0.3))  # Up to +30% for tables
        
        return weight
    
    def _find_heading_context(self, chunk: str, headings: List[str]) -> str:
        """Find the most relevant heading for this chunk"""
        if not headings:
            return ""
        
        chunk_lower = chunk.lower()
        best_heading = ""
        best_score = 0
        
        for heading in headings:
            heading_lower = heading.lower()
            
            # Simple keyword overlap scoring
            heading_words = set(re.findall(r'\b\w+\b', heading_lower))
            chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
            
            overlap = len(heading_words & chunk_words)
            if overlap > best_score:
                best_score = overlap
                best_heading = heading
        
        return best_heading


# ============================================================================
# Test/Demo Code
# ============================================================================

if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        chunk_size: int = 500
        chunk_overlap: int = 50
        max_passages_per_task: int = 3
    
    # Test data
    sample_page = WebPage(
        url="https://example.com/test",
        title="Test Page About Machine Learning",
        content="""
        Machine learning is a branch of artificial intelligence. 
        It focuses on the use of data and algorithms to imitate the way that humans learn.
        Deep learning is a subset of machine learning. Neural networks are fundamental to deep learning.
        Applications include computer vision, natural language processing, and recommendation systems.
        """,
        headings=["Introduction", "Deep Learning", "Applications"]
    )
    
    sample_task = PlanTask(
        fact="What is machine learning?",
        variable_name="ml_definition",
        category="background"
    )
    
    sample_plan = ExtractionPlan(
        archetype="background",
        entity="machine learning",
        tasks_to_extract=[sample_task]
    )
    
    config = MockConfig()
    ranker = ContentRanker(config)
    
    print("[TEST][Ranker] Testing content ranking...")
    print("=" * 80)
    
    task_passages = ranker.rank_passages(sample_plan, [sample_page])
    
    print("\n" + "=" * 80)
    print(f"[TEST][Result] Ranked passages for {len(task_passages)} tasks")
    print("\n[TEST][Passages]:")
    for var_name, passages in task_passages.items():
        print(f"\n  Task: {var_name}")
        for i, passage in enumerate(passages, 1):
            print(f"    {i}. Score: {passage.score:.2f}")
            print(f"       Text: {passage.text[:100]}...")
            print(f"       Heading: {passage.heading_context}")
    print("=" * 80)


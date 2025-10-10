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
    source_url: str
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
        
        # Split content into chunks
        chunks = self._chunk_content(page.content)
        
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
                source_url=source_url,
                heading_context=heading_context,
                score=score,
                position=i,
                metadata=page_meta
            )
            
            passages.append(passage)

        return passages
    
    def _chunk_content(self, content: str) -> List[str]:
        """Split content into overlapping chunks"""
        sentences = re.split(r'[.!?]+', content)
        chunks = []
        
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.config.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk.split('.')[-2:]  # Keep last 2 sentences for overlap
                current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                current_length = len(current_chunk.split())
            else:
                current_chunk += '. ' + sentence if current_chunk else sentence
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _calculate_relevance_score(self, chunk: str, task: PlanTask, page: WebPage) -> float:
        """Calculate relevance score for a chunk"""
        score = 0.0
        chunk_lower = chunk.lower()
        task_lower = task.fact.lower()
        
        # Keyword matching
        task_keywords = re.findall(r'\b\w+\b', task_lower)
        for keyword in task_keywords:
            if len(keyword) > 3:  # Skip short words
                if keyword in chunk_lower:
                    score += 1.0
        
        # Entity matching (if available)
        if hasattr(task, 'entity') and task.entity:
            entity_lower = task.entity.lower()
            if entity_lower in chunk_lower:
                score += 2.0
        
        # Question word matching
        question_words = ['what', 'when', 'where', 'who', 'why', 'how']
        for qword in question_words:
            if qword in task_lower and qword in chunk_lower:
                score += 0.5
        
        # Length penalty for very short chunks
        if len(chunk.split()) < 20:
            score *= 0.5
        
        # Boost for structured content (lists, numbers, dates)
        if re.search(r'\d{4}', chunk):  # Years
            score += 0.3
        if re.search(r'\b\d+[.,]\d+\b', chunk):  # Numbers
            score += 0.2
        if re.search(r'^\s*[-•*]\s', chunk, re.MULTILINE):  # Lists
            score += 0.2
        
        # Apply source-based weighting
        source_weight = self._get_source_weight(page)
        score *= source_weight
        
        return score
    
    def _get_source_weight(self, page: WebPage) -> float:
        """
        Calculate source weight based on content type and quality indicators.
        Uses dynamic heuristics, not hardcoded domain lists.
        """
        metadata = page.metadata or {}
        
        # Base weight = 1.0
        weight = 1.0
        
        # Factor 1: Content type priority
        source_type = metadata.get('source_type', 'unknown')
        extraction_type = metadata.get('extraction_type', 'standard')
        
        if source_type == 'snippet':
            # Snippets are good for Answer Boxes but lack context
            weight = 1.0
        elif source_type == 'webpage':
            # Full web pages have more context
            weight = 1.2
            
            # Factor 2: Content richness (Wikipedia-like, Baike-like)
            if extraction_type in ('wikipedia', 'wikipedia_optimized'):
                weight = 1.5  # Wikipedia optimized content
            elif extraction_type == 'pdf':
                weight = 1.3  # PDF documents (usually official/academic)
        
        # Factor 3: Content quality indicators
        content_length = len(page.content.split())
        if content_length > 500:  # Rich content
            weight *= 1.1
        elif content_length < 100:  # Too short
            weight *= 0.8
        
        # Factor 4: Structured content indicators (infobox, sections)
        if metadata.get('has_infobox'):
            weight *= 1.2
        if metadata.get('section_count', 0) > 5:  # Well-structured article
            weight *= 1.1
        
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


"""
Information Retrieval and Reasoning Action for WebChasor
Handles factual queries that require external information retrieval.
"""

import os
import json
import asyncio
import logging
import requests
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import time
from utils.timectx import TimeContext, parse_time_intent
from artifacts import Action, Context, Artifact
from actions.serpapi_search import SerpAPISearch
from actions.querymaker import QueryMaker
from config_manager import get_config

# Setup logging
logger = logging.getLogger(__name__)

class RetrievalProvider(Enum):
    """Available retrieval providers"""
    SERPAPI = "serpapi"
    VECTOR_DB = "vector_db"  # Placeholder for future implementation
    HYBRID = "hybrid"        # Combine both sources

@dataclass
class PlanTask:
    """Individual extraction task from planner"""
    fact: str
    variable_name: str
    category: str = "other"
    confidence_threshold: float = 0.7

@dataclass
class ExtractionPlan:
    """Complete plan from planner"""
    archetype: str = "general"
    entity: Optional[str] = None
    tasks_to_extract: List[PlanTask] = field(default_factory=list)
    final_calculation: Optional[Dict[str, Any]] = None
    presentation_hint: Optional[Dict[str, Any]] = None

@dataclass
class SearchResult:
    """Structured search result"""
    title: str
    snippet: str
    url: str
    source: str
    result_type: str = "organic"  # organic, answer_box, knowledge_graph, news
    position: int = 0
    date: Optional[str] = None
    confidence: float = 0.0

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

@dataclass
class ContentPassage:
    """Ranked content passage for extraction"""
    text: str
    source_url: str
    heading_context: str = ""
    score: float = 0.0
    task_relevance: Dict[str, float] = field(default_factory=dict)
    position: int = 0

@dataclass
class ExtractedVariable:
    """Extracted information for a specific variable"""
    variable_name: str
    value: Any
    confidence: float
    provenance: List[str] = field(default_factory=list)  # Source URLs
    extraction_method: str = "llm"  # llm, regex, ner
    raw_passages: List[str] = field(default_factory=list)

@dataclass
class IRConfig:
    """Configuration for IR_RAG execution"""
    # Search configuration
    max_search_results: int = 10
    search_provider: RetrievalProvider = RetrievalProvider.SERPAPI
    search_location: str = "Hong Kong"
    search_language: str = "zh-cn"
    
    # Content fetching
    enable_web_scraping: bool = False  # 🔧 NEW: Web scraping toggle
    max_pages_to_visit: int = 5
    fetch_timeout: int = 10
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_passage_length: int = 20  # Added missing field
    
    # Extraction configuration
    confidence_threshold: float = 0.7  # Renamed to match config
    max_passages_per_task: int = 3
    max_extraction_attempts: int = 3  # Added missing field
    
    # Vector DB configuration (placeholder)
    vector_db_enabled: bool = False
    vector_top_k: int = 10
    hybrid_weight: float = 0.5  # Weight for combining vector and search results

class Planner:
    """Planner component that creates extraction plans"""
    
    def __init__(self, llm_client=None, model_name: str = None):
        """Initialize planner with LLM client"""
        self.client = llm_client
        self.model_name = model_name or get_config().get('models.planner.model_name', 'gpt-4')
        # Import planner prompt
        try:
            from prompt import PLANNER_PROMPT
            self.planner_prompt = PLANNER_PROMPT
        except ImportError:
            logger.warning("Could not import PLANNER_PROMPT, using fallback")
            self.planner_prompt = self._get_fallback_prompt()
    
    def _get_fallback_prompt(self) -> str:
        """Fallback planner prompt if import fails"""
        return """
        You are a Task Planner. Decompose the user query into structured extraction tasks.
        Output MUST be valid JSON with this schema:
        {
          "plan": {
            "archetype": "biography|fact_verification|recent_situation|background|comparison|other",
            "entity": "main entity name",
            "tasks_to_extract": [
              {
                "fact": "What specific fact to extract?",
                "variable_name": "snake_case_name",
                "category": "biography|fact_verification|recent_situation|background|comparison|other"
              }
            ]
          }
        }
        """
    
    async def plan(self, query: str, archetype_hint: Optional[str] = None) -> ExtractionPlan:
        """Create extraction plan for the given query"""
        try:
            print(f"📋 PLANNER: Creating plan for query: {query[:100]}...")
            
            if not self.client:
                logger.warning("No LLM client available, using fallback plan")
                return self._create_fallback_plan(query)
            
            # Build planner prompt with query
            prompt = self.planner_prompt.replace("{user_query}", query)
            
            # Call LLM for planning
            cfg = get_config()
            planner_max_tokens = cfg.get('models.planner.max_tokens', 2000)
            planner_temperature = cfg.get('models.planner.temperature', 0.1)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"User: {query}"}
                ],
                temperature=planner_temperature,
                max_tokens=planner_max_tokens
            )
            
            plan_text = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason

            # Check if response was truncated
            if finish_reason == 'length':
                logger.warning(f"Planner response was truncated (finish_reason=length), may cause JSON parse error")
                print(f"[PLANNER][WARN] Response truncated at {len(plan_text)} chars, consider increasing max_tokens")
            
            # Parse JSON response
            try:
                plan_data = json.loads(plan_text)
                plan_dict = plan_data.get("plan", plan_data)
                
                # Convert to structured plan
                tasks = []
                for task_data in plan_dict.get("tasks_to_extract", []):
                    tasks.append(PlanTask(
                        fact=task_data["fact"],
                        variable_name=task_data["variable_name"],
                        category=task_data.get("category", "other")
                    ))
                
                extraction_plan = ExtractionPlan(
                    archetype=plan_dict.get("archetype", "general"),
                    entity=plan_dict.get("entity"),
                    tasks_to_extract=tasks,
                    final_calculation=plan_dict.get("final_calculation"),
                    presentation_hint=plan_dict.get("presentation_hint")
                )
                
                print(f"📋 PLANNER: Created plan with {len(tasks)} tasks, archetype: {extraction_plan.archetype}")
                return extraction_plan
                
            except json.JSONDecodeError as e:
                print(f"[PLANNER][ERROR] Failed to parse planner JSON: {e}")
                print(f"[PLANNER][ERROR] Raw response (first 500 chars): {plan_text[:500]}")
                print(f"[PLANNER][ERROR] Raw response (last 200 chars): {plan_text[-200:]}")
                logger.error(f"Failed to parse planner JSON: {e}")
                logger.error(f"Raw response: {plan_text}")
                print(f"[PLANNER][FALLBACK] Using fallback plan")
                return self._create_fallback_plan(query)
                
        except Exception as e:
            logger.error(f"Planner error: {e}")
            return self._create_fallback_plan(query)
    
    def _create_fallback_plan(self, query: str) -> ExtractionPlan:
        """Create a simple fallback plan when LLM planning fails"""
        # Extract key terms for basic planning - support both English and Chinese
        english_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        chinese_terms = re.findall(r'[\u4e00-\u9fff]+', query)
        
        # Prefer Chinese terms for Chinese queries, English terms for English queries
        if chinese_terms:
            entity = chinese_terms[0]  # Use first Chinese term
        elif english_terms:
            entity = english_terms[0]  # Use first English term
        else:
            entity = "unknown"
        
        # Create basic extraction tasks with appropriate language
        if chinese_terms:
            # Use Chinese for Chinese queries
            tasks = [
                PlanTask(
                    fact=f"关于{entity}的基本信息是什么？",
                    variable_name="basic_info",
                    category="background"
                ),
                PlanTask(
                    fact=f"关于{entity}的最新信息有哪些？",
                    variable_name="key_facts",
                    category="recent_situation"
                )
            ]
        else:
            # Use English for English queries
            tasks = [
                PlanTask(
                    fact=f"What is the basic information about {entity}?",
                    variable_name="basic_info",
                    category="background"
                ),
                PlanTask(
                    fact=f"What are the recent facts about {entity}?",
                    variable_name="key_facts",
                    category="recent_situation"
                )
            ]
        
        return ExtractionPlan(
            archetype="general",
            entity=entity,
            tasks_to_extract=tasks
        )

class WebVisitor:
    """Component for fetching and cleaning web pages"""
    
    def __init__(self, config: IRConfig):
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
        
        print(f"🌐 VISITOR: Successfully fetched {len(successful_pages)}/{len(urls)} pages")
        return successful_pages
    
    async def _fetch_single(self, url: str) -> WebPage:
        """Fetch and clean a single web page"""
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
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
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
            
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return WebPage(
                url=url, title="", content="", success=False,
                error=str(e), fetch_time=time.time() - start_time
            )
    
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

class ContentRanker:
    """Component for ranking and chunking content passages"""
    
    def __init__(self, config: IRConfig):
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
            
            passage = ContentPassage(
                text=chunk,
                source_url=page.url,
                heading_context=heading_context,
                score=score,
                position=i
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
        
        return score
    
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

class InformationExtractor:
    """Component for extracting structured information from passages"""
    
    def __init__(self, llm_client=None, model_name: str = None):
        self.client = llm_client
        self.model_name = model_name or os.getenv("OPENAI_API_MODEL_AGENT_SYNTHESIZER", "gpt-3.5-turbo")
    
    async def extract_variables(self, plan: ExtractionPlan, ranked_passages: Dict[str, List[ContentPassage]]) -> Dict[str, ExtractedVariable]:
        """Extract variables from ranked passages using LLM"""
        print(f"🔍 EXTRACTOR: Extracting {len(plan.tasks_to_extract)} variables...")
        
        extracted_vars = {}
        
        for task in plan.tasks_to_extract:
            passages = ranked_passages.get(task.variable_name, [])
            if not passages:
                print(f"🔍 EXTRACTOR: No passages found for {task.variable_name}")
                continue
            
            # Extract variable using LLM or fallback methods
            extracted_var = await self._extract_single_variable(task, passages)
            extracted_vars[task.variable_name] = extracted_var
            
            print(f"🔍 EXTRACTOR: Extracted '{task.variable_name}' with confidence {extracted_var.confidence:.2f}")
        
        return extracted_vars
    
    async def _extract_single_variable(self, task: PlanTask, passages: List[ContentPassage]) -> ExtractedVariable:
        """Extract a single variable from passages"""
        try:
            if self.client:
                return await self._llm_extract(task, passages)
            else:
                return self._fallback_extract(task, passages)
        except Exception as e:
            logger.error(f"Extraction failed for {task.variable_name}: {e}")
            return self._fallback_extract(task, passages)
    
    async def _llm_extract(self, task: PlanTask, passages: List[ContentPassage]) -> ExtractedVariable:
        """Use LLM to extract information"""
        # Combine top passages
        combined_text = "\n\n".join([f"Source: {p.source_url}\n{p.text}" for p in passages[:3]])
        
        # Build extraction prompt
        prompt = f"""
        Extract the following information from the provided text passages:
        
        QUESTION: {task.fact}
        VARIABLE: {task.variable_name}
        CATEGORY: {task.category}
        
        TEXT PASSAGES:
        {combined_text}
        
        Please provide a JSON response with:
        {{
            "value": "extracted information",
            "confidence": 0.8,
            "reasoning": "brief explanation of why this answer is correct"
        }}
        
        If the information is not found, set confidence to 0.0 and value to null.
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert information extractor. Extract precise, factual information from text passages."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        try:
            result_data = json.loads(result_text)
            return ExtractedVariable(
                variable_name=task.variable_name,
                value=result_data.get("value"),
                confidence=float(result_data.get("confidence", 0.0)),
                provenance=[p.source_url for p in passages],
                extraction_method="llm",
                raw_passages=[p.text for p in passages]
            )
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return ExtractedVariable(
                variable_name=task.variable_name,
                value=result_text,
                confidence=0.5,
                provenance=[p.source_url for p in passages],
                extraction_method="llm_fallback",
                raw_passages=[p.text for p in passages]
            )
    
    def _fallback_extract(self, task: PlanTask, passages: List[ContentPassage]) -> ExtractedVariable:
        """Fallback extraction using simple heuristics"""
        if not passages:
            return ExtractedVariable(
                variable_name=task.variable_name,
                value=None,
                confidence=0.0,
                extraction_method="fallback"
            )
        
        # Use the highest-scored passage as the answer
        best_passage = passages[0]
        
        # Simple extraction based on task category
        if task.category == "biography":
            value = self._extract_biographical_info(best_passage.text)
        elif task.category == "fact_verification":
            value = self._extract_factual_claim(best_passage.text)
        else:
            # Generic extraction - use first sentence or paragraph
            sentences = best_passage.text.split('.')
            value = sentences[0].strip() if sentences else best_passage.text[:200]
        
        return ExtractedVariable(
            variable_name=task.variable_name,
            value=value,
            confidence=0.6,
            provenance=[p.source_url for p in passages],
            extraction_method="fallback",
            raw_passages=[p.text for p in passages]
        )
    
    def _extract_biographical_info(self, text: str) -> str:
        """Extract biographical information"""
        # Look for patterns like "born in", "graduated from", etc.
        bio_patterns = [
            r'born (?:in|on) ([^.]+)',
            r'graduated from ([^.]+)',
            r'worked at ([^.]+)',
            r'known for ([^.]+)'
        ]
        
        for pattern in bio_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback to first sentence
        sentences = text.split('.')
        return sentences[0].strip() if sentences else text[:200]
    
    def _extract_factual_claim(self, text: str) -> str:
        """Extract factual claims"""
        # Look for definitive statements
        fact_patterns = [
            r'(was founded in \d{4})',
            r'(established in \d{4})',
            r'(created in \d{4})',
            r'(\d{4}[^.]*founded)'
        ]
        
        for pattern in fact_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback to first sentence
        sentences = text.split('.')
        return sentences[0].strip() if sentences else text[:200]

class IR_RAG(Action):
    """
    Information Retrieval and Reasoning Action
    
    Handles queries that require external information retrieval and synthesis.
    Workflow: Planning → Retrieval → Reading → Ranking → Extraction → Synthesis
    """
    
    name = "IR_RAG"
    requires_tools = ["search", "synthesizer"]
    max_time_s = 120
    max_tokens_in = 10000
    max_tokens_out = 2000
    
    def __init__(self, config: Optional[IRConfig] = None, llm_client=None):
        """Initialize IR_RAG with configuration and LLM client"""
        self.config = config or self._load_config_from_yaml()
        self.llm_client = llm_client
        
        # Initialize components with config-driven model names
        cfg = get_config()
        planner_model = cfg.get('models.planner.model_name', 'gpt-4')
        extractor_model = cfg.get('models.extractor.model_name', 'gpt-3.5-turbo')
        
        self.planner = Planner(llm_client, planner_model)
        self.querymaker = QueryMaker()
        self.search_tool = SerpAPISearch()
        self.visitor = WebVisitor(self.config)
        self.ranker = ContentRanker(self.config)
        self.extractor = InformationExtractor(llm_client, extractor_model)
        
        if cfg.is_decision_logging_enabled('ir_rag'):
            print(f"[IR_RAG][INIT] provider={self.config.search_provider.value} max_pages={self.config.max_pages_to_visit} web_scraping={'ON' if self.config.enable_web_scraping else 'OFF'}")
    
    def _load_config_from_yaml(self) -> IRConfig:
        """Load IR_RAG configuration from YAML config file"""
        cfg = get_config()
        ir_config = cfg.get_section('ir_rag')
        
        return IRConfig(
            search_provider=RetrievalProvider(ir_config.get('search', {}).get('provider', 'serpapi')),
            max_search_results=ir_config.get('search', {}).get('max_results', 10),
            max_pages_to_visit=ir_config.get('web_scraping', {}).get('max_pages', 5),
            search_location=ir_config.get('search', {}).get('location', 'Hong Kong'),
            search_language=ir_config.get('search', {}).get('language', 'zh-cn'),
            enable_web_scraping=ir_config.get('web_scraping', {}).get('enabled', False),
            chunk_size=ir_config.get('content', {}).get('chunk_size', 500),
            chunk_overlap=ir_config.get('content', {}).get('chunk_overlap', 50),
            max_passages_per_task=ir_config.get('content', {}).get('max_passages_per_task', 3),
            min_passage_length=ir_config.get('content', {}).get('min_passage_length', 20),
            confidence_threshold=ir_config.get('extraction', {}).get('confidence_threshold', 0.7),
            max_extraction_attempts=ir_config.get('extraction', {}).get('max_extraction_attempts', 3)
        )
    
    async def run(self, ctx: Context, toolset) -> Artifact:
        """
        Execute the complete IR workflow
        
        Args:
            ctx: Context containing query and metadata
            toolset: Available tools including synthesizer
            
        Returns:
            Artifact: Synthesized information with citations
        """
        try:
            print(f"[IR_RAG][START] query={ctx.query[:100]}...")
            
            # Step 1: Planning - Create extraction plan
            plan = await self.planner.plan(ctx.query)
            print(f"[IR_RAG][PLAN] tasks={len(plan.tasks_to_extract)}")
            
            # Step 2: Retrieval - Search for information
            search_results = await self._retrieve_information(ctx.query, plan, ctx)
            print(f"[IR_RAG][SEARCH] results={search_results}")
            print(f"[IR_RAG][SEARCH] results={len(search_results)}")
            
            if not search_results:
                return self._create_no_results_artifact(ctx.query)
            
            # Step 3: Reading - Conditional web scraping
            if self.config.enable_web_scraping:
                print(f"[IR_RAG][WEB] scraping=ENABLED fetching pages...")
                selected_urls = self._select_seed_urls(search_results)
                pages = await self.visitor.fetch_many(selected_urls)
                print(f"[IR_RAG][WEB] fetched={len(pages)} pages")
                
                if not pages:
                    print(f"[IR_RAG][WARN] No pages fetched, fallback to snippets")
                    pages = self._create_pages_from_snippets(search_results)
            else:
                print(f"[IR_RAG][WEB] scraping=DISABLED using snippets only")
                pages = self._create_pages_from_snippets(search_results)
            
            # Step 4: Ranking - Rank content passages by relevance
            ranked_passages = self.ranker.rank_passages(plan, pages)
            
            # Step 5: Extraction - Extract structured information
            extracted_vars = await self.extractor.extract_variables(plan, ranked_passages)
            print(f"[IR_RAG][EXTRACT] variables={len(extracted_vars)}")
            
            # Step 6: Synthesis - Generate final response
            artifact = await self._synthesize_response(ctx, plan, extracted_vars, search_results, toolset)
            
            print(f"[IR_RAG][DONE] workflow completed")
            return artifact
            
        except Exception as e:
            logger.error(f"IR_RAG workflow failed: {e}")
            return self._create_error_artifact(ctx.query, str(e))
    
    async def _retrieve_information(self, query: str, plan: ExtractionPlan, ctx: Context) -> List[SearchResult]:
        """Retrieve information using configured search provider"""
        
        if self.config.search_provider == RetrievalProvider.SERPAPI:
            return await self._serpapi_search(query, plan, ctx)
        elif self.config.search_provider == RetrievalProvider.VECTOR_DB:
            return await self._vector_search(query, plan)
        elif self.config.search_provider == RetrievalProvider.HYBRID:
            return await self._hybrid_search(query, plan, ctx)
        else:
            raise ValueError(f"Unsupported search provider: {self.config.search_provider}")
    
    async def _serpapi_search(self, query: str, plan: ExtractionPlan, ctx: Context) -> List[SearchResult]:
        """Search using SerpAPI with multiple diverse queries"""
        print(f"🔍 IR_RAG: Searching with SerpAPI...")

        # Build multiple search queries from plan
        search_queries = self._build_search_query(query, plan, ctx)
        
        print(f"🔍 IR_RAG: Generated {len(search_queries)} search queries:")
        for i, q in enumerate(search_queries, 1):
            print(f"  {i}. {q}")
        print(f"🔍 IR_RAG: Time intent: {ctx.time_context.intent if ctx.time_context else 'None'}")
        
        # Execute search for each query and collect results
        all_search_results = []
        seen_urls = set()  # Deduplicate by URL
        
        for search_query in search_queries:
            try:
                # Execute search using structured results method
                raw_results = self.search_tool.get_structured_results(
                    query=search_query,
                    num_results=self.config.max_search_results,
                    location=self.config.search_location,
                    language=self.config.search_language
                )
                
                # Convert raw results to SearchResult objects and deduplicate
                for result in raw_results:
                    url = result.get('link', '')
                    
                    # Skip if we've already seen this URL
                    if url in seen_urls:
                        continue
                    
                    seen_urls.add(url)
                    
                    search_result = SearchResult(
                        title=result.get('title', ''),
                        snippet=result.get('snippet', ''),
                        url=url,
                        source=result.get('source', ''),
                        result_type=result.get('type', 'organic'),
                        position=result.get('position', 0),
                        date=result.get('date'),
                        confidence=0.8  # Default confidence for SerpAPI results
                    )
                    all_search_results.append(search_result)
            
            except Exception as e:
                logger.error(f"[IR_RAG][SEARCH] Error searching with query '{search_query}': {e}")
                continue
        
        print(f"🔍 IR_RAG: Retrieved {len(all_search_results)} unique results from {len(search_queries)} queries")
        return all_search_results
    
    async def _vector_search(self, query: str, plan: ExtractionPlan) -> List[SearchResult]:
        """Search using vector database (placeholder)"""
        print(f"🔍 IR_RAG: Vector search not implemented yet, using fallback...")
        
        # Placeholder for future vector database integration
        # This would query a vector database with embeddings
        
        return []
    
    async def _hybrid_search(self, query: str, plan: ExtractionPlan, ctx: Context) -> List[SearchResult]:
        """Combine SerpAPI and vector search results"""
        print(f"🔍 IR_RAG: Hybrid search not fully implemented, using SerpAPI only...")
        
        # For now, just use SerpAPI
        # In the future, this would combine results from both sources
        return await self._serpapi_search(query, plan, ctx)
    
    def _build_search_query(self, original_query: str, plan: ExtractionPlan, ctx: Context) -> List[str]:
        """
        Build optimized search queries using LLM-based QueryMaker.
        
        Args:
            original_query: User's original query
            plan: Extraction plan from planner
            ctx: Context with time and other information
            
        Returns:
            List of diverse search queries
        """
        try:
            # Use QueryMaker to generate diverse queries
            queries = self.querymaker.generate_queries(original_query, ctx)
            logger.info(f"[IR_RAG][QUERY] Generated {len(queries)} search queries")
            return queries
        except Exception as e:
            logger.error(f"[IR_RAG][QUERY] QueryMaker failed: {e}, using fallback")
            # Fallback to original query
            return [original_query]
    
    def _parse_serpapi_results(self, raw_results: str, query: str) -> List[SearchResult]:
        """Parse SerpAPI formatted results into structured format"""
        # This is a simplified parser - in production you'd want more robust parsing
        results = []
        
        if "No results found" in raw_results or "Error:" in raw_results:
            return results
        
        # Extract results using regex patterns
        # This is a basic implementation - you might want to modify SerpAPISearch 
        # to return structured data instead
        
        # Look for title and URL patterns
        url_pattern = r'📎 Link: (https?://[^\s\n]+)'
        title_pattern = r'\*\*(.*?)\*\*'
        
        urls = re.findall(url_pattern, raw_results)
        titles = re.findall(title_pattern, raw_results)
        
        # Create search results
        for i, (title, url) in enumerate(zip(titles, urls)):
            if url and title:
                results.append(SearchResult(
                    title=title,
                    snippet="",  # Would need more parsing to extract snippets
                    url=url,
                    source=urlparse(url).netloc,
                    position=i + 1,
                    confidence=0.8
                ))
        
        print(f"🔍 IR_RAG: Parsed {len(results)} structured results from SerpAPI")
        return results
    
    def _select_seed_urls(self, search_results: List[SearchResult]) -> List[str]:
        """Select URLs to visit based on diversity and authority"""
        if not search_results:
            return []
        
        selected_urls = []
        seen_domains = set()
        
        # Prioritize different domains for diversity
        for result in search_results:
            domain = urlparse(result.url).netloc
            
            # Skip if we already have a URL from this domain (unless it's a high-authority domain)
            if domain in seen_domains and domain not in ['wikipedia.org', 'britannica.com', 'reuters.com']:
                continue
            
            selected_urls.append(result.url)
            seen_domains.add(domain)
            
            # Stop when we have enough URLs
            if len(selected_urls) >= self.config.max_pages_to_visit:
                break
        
        return selected_urls
    
    def _create_pages_from_snippets(self, search_results: List[SearchResult]) -> List[WebPage]:
        """Create dummy WebPage objects from search results if web scraping is disabled"""
        print(f"🌐 IR_RAG: Creating dummy pages from search snippets for ranking...")
        pages = []
        for i, result in enumerate(search_results):
            pages.append(WebPage(
                url=result.url,
                title=result.title,
                content=result.snippet,
                headings=[],
                metadata={'status_code': 200, 'content_type': 'text/html'},
                fetch_time=0.0,
                success=True
            ))
        return pages

    async def _synthesize_response(self, ctx: Context, plan: ExtractionPlan, 
                                 extracted_vars: Dict[str, ExtractedVariable], 
                                 search_results: List[SearchResult], toolset) -> Artifact:
        """Synthesize final response using extracted information"""
        
        # Build materials for synthesis
        materials = self._build_synthesis_materials(plan, extracted_vars, search_results)
        
        # Create synthesis prompt
        synthesis_prompt = f"""
        Based on the extracted information, provide a comprehensive answer to: {ctx.query}.MUST use all retrieved information, no missing information. Be as comprehensive as possible. Add background information if there is any possible.
        
        Use the following extracted information:
        {materials}
        
        Please provide a well-structured response with:
        1. Clear sections based on the information found
        2. Citations using [1], [2], etc. format
        3. Factual accuracy based on the sources
        4. Time context: {ctx.time_context.intent}. Notice the time context is important. Filter the information based on the time context.
        5. With a reference list and urls corresponding to the citations.
        6. MUST use all retrieved information, no missing information.
        """
        

        # Call synthesizer using the new generate method for proper global prompt support
        if hasattr(toolset, 'synthesizer'):
            # Set up constraints for IR_RAG synthesis with temperature from config
            cfg = get_config()
            ir_rag_temperature = cfg.get('models.synthesizer.temperature', 0.5)
            
            constraints = {
                "language": "auto",  # Let synthesizer detect language
                "tone": "factual, authoritative",
                "temperature": ir_rag_temperature,  # Use config value
                "instruction_hint": "Provide a comprehensive answer with clear sections and proper citations using [1], [2], etc. format."
            }
            
            response = await toolset.synthesizer.generate(
                category="INFORMATION_RETRIEVAL",
                style_key="auto",  # 使用 auto 启用自动语言检测和样式选择
                constraints=constraints,
                materials=f"Query: {ctx.query}\n\nExtracted Information:\n{materials}",
                task_scaffold=None
            )
        else:
            response = self._create_fallback_response(ctx.query, extracted_vars)
        # Build citations
        citations = self._build_citations(extracted_vars, search_results)
        
        # Create artifact
        artifact = Artifact(
            kind="text",
            content=response,
            meta={
                'query': ctx.query,
                'plan': plan,
                'extracted_variables': {k: v.value for k, v in extracted_vars.items()},
                'citations': citations,
                'search_results_count': len(search_results),
                'extraction_confidence': self._calculate_overall_confidence(extracted_vars),
                'sources': list(set([url for var in extracted_vars.values() for url in var.provenance]))
            }
        )
        
        return artifact
    
    def _build_synthesis_materials(self, plan: ExtractionPlan, 
                                 extracted_vars: Dict[str, ExtractedVariable], 
                                 search_results: List[SearchResult]) -> str:
        """Build materials string for synthesis"""
        materials = []
        citation_counter = 1
        
        # Add extracted variables with numbered citations
        for var_name, var_data in extracted_vars.items():
            if var_data.value and var_data.confidence > 0.3:
                # Use numbered citation instead of variable name
                materials.append(f"[{citation_counter}] **{var_name}**: {var_data.value}")
                citation_counter += 1
        
        # Add high-confidence search results (continue numbering)
        for result in search_results[:3]:
            if result.snippet:
                materials.append(f"[{citation_counter}] {result.title}: {result.snippet}")
                citation_counter += 1
        
        return "\n\n".join(materials)
    
    def _build_citations(self, extracted_vars: Dict[str, ExtractedVariable], 
                        search_results: List[SearchResult]) -> List[Dict[str, str]]:
        """Build citation list"""
        citations = []
        seen_urls = set()
        
        # Add sources from extracted variables
        for var in extracted_vars.values():
            for url in var.provenance:
                if url not in seen_urls:
                    # Find corresponding search result for title
                    title = url
                    for result in search_results:
                        if result.url == url:
                            title = result.title
                            break
                    
                    citations.append({
                        'title': title,
                        'url': url,
                        'source': urlparse(url).netloc
                    })
                    seen_urls.add(url)
        
        return citations
    
    def _calculate_overall_confidence(self, extracted_vars: Dict[str, ExtractedVariable]) -> float:
        """Calculate overall confidence score"""
        if not extracted_vars:
            return 0.0
        
        confidences = [var.confidence for var in extracted_vars.values()]
        return sum(confidences) / len(confidences)
    
    def _create_fallback_response(self, query: str, extracted_vars: Dict[str, ExtractedVariable]) -> str:
        """Create fallback response when synthesizer is not available"""
        response_parts = [f"Based on the available information, here's what I found about: {query}\n"]
        
        for var_name, var_data in extracted_vars.items():
            if var_data.value:
                response_parts.append(f"• **{var_name.replace('_', ' ').title()}**: {var_data.value}")
        
        if not any(var.value for var in extracted_vars.values()):
            response_parts.append("Unfortunately, I couldn't find specific information to answer your query.")
        
        return "\n".join(response_parts)
    
    def _create_no_results_artifact(self, query: str) -> Artifact:
        """Create artifact when no search results are found"""
        return Artifact(
            kind="text",
            content=f"I couldn't find any relevant information for your query: {query}. Please try rephrasing your question or checking for typos.",
            meta={
                'query': query,
                'error': 'no_search_results',
                'search_results_count': 0
            }
        )
    
    def _create_fetch_failed_artifact(self, query: str, search_results: List[SearchResult]) -> Artifact:
        """Create artifact when page fetching fails"""
        return Artifact(
            kind="text",
            content=f"I found search results for your query but couldn't access the web pages. Here are the search results I found:\n\n" +
                   "\n".join([f"• {result.title}: {result.url}" for result in search_results[:5]]),
            meta={
                'query': query,
                'error': 'fetch_failed',
                'search_results_count': len(search_results)
            }
        )
    
    def _create_error_artifact(self, query: str, error: str) -> Artifact:
        """Create artifact for general errors"""
        return Artifact(
            kind="text",
            content=f"I encountered an error while processing your query: {query}. Error: {error}",
            meta={
                'query': query,
                'error': error
            }
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of IR_RAG components"""
        health = {
            'status': 'healthy',
            'components': {},
            'config': {
                'search_provider': self.config.search_provider.value,
                'max_pages': self.config.max_pages_to_visit,
                'vector_db_enabled': self.config.vector_db_enabled
            }
        }
        
        # Check SerpAPI
        try:
            if self.search_tool.api_key:
                health['components']['serpapi'] = 'configured'
            else:
                health['components']['serpapi'] = 'missing_api_key'
        except Exception as e:
            health['components']['serpapi'] = f'error: {e}'
        
        # Check LLM client
        if self.llm_client:
            health['components']['llm'] = 'available'
        else:
            health['components']['llm'] = 'not_configured'
        
        return health
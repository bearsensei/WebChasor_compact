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
from actions.google_search import GoogleCustomSearch
from actions.gcp_vertex_search import GCPVertexSearch
from actions.querymaker import QueryMaker
from config_manager import get_config

# Setup logging
logger = logging.getLogger(__name__)

class RetrievalProvider(Enum):
    """Available retrieval providers"""
    SERPAPI = "serpapi"
    GOOGLE_CUSTOM_SEARCH = "google_custom_search"
    GCP_VERTEX_SEARCH = "gcp_vertex_search"
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
    search_providers: List[RetrievalProvider] = None  # Multiple providers for hybrid/fallback
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
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Chinese or English"""
        if not text:
            return "en"
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.strip())
        if total_chars == 0:
            return "en"
        return "zh" if chinese_chars / total_chars > 0.3 else "en"
    
    def _extract_core_entities(self, question: str) -> List[str]:
        """
        Extract core entities from the question for validation.
        Simple heuristic-based extraction (can be enhanced with NER later).
        """
        # Remove common question words
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 
                         '什么', '何时', '哪里', '谁', '为什么', '怎么', '哪个', '多少']
        
        # Tokenize and filter
        import jieba
        words = list(jieba.cut(question))
        
        # Filter out question words, punctuation, and short words
        entities = []
        for word in words:
            word_lower = word.lower().strip()
            if (len(word) >= 2 and 
                word_lower not in question_words and 
                not word.strip() in '，。？！、；：""''（）【】《》' and
                not word.isdigit()):
                entities.append(word)
        
        # Return top entities (limit to avoid too many)
        return entities[:5]
    
    async def _llm_extract(self, task: PlanTask, passages: List[ContentPassage]) -> ExtractedVariable:
        """Use LLM to extract information"""
        # Get max passages from config (default to 20 if not set)
        cfg = get_config()
        max_passages = cfg.get('ir_rag.content.max_passages_per_task', 20)
        
        # Combine top passages (use config value)
        combined_text = "\n\n".join([f"Source {i+1}: {p.source_url}\n{p.text}" for i, p in enumerate(passages[:max_passages])])
        
        # Extract core entities from the question for validation
        core_entities = self._extract_core_entities(task.fact)
        entities_str = ", ".join(core_entities) if core_entities else "N/A"
        
        # Detect language and build appropriate prompt
        lang = self._detect_language(task.fact + " " + combined_text[:500])
        
        if lang == "zh":
            # Chinese prompt
            prompt = f"""从提供的文本段落中提取以下信息：

问题: {task.fact}
变量名: {task.variable_name}
类别: {task.category}
核心实体: {entities_str}

文本段落:
{combined_text}

**关键要求**:
1. 仅从提供的文本段落中提取信息
2. 不要使用你自己的知识或做假设
3. 提取的信息必须与核心实体相关
4. 如果多个段落提供不同信息，优先选择最具体和最新的
5. 尽可能引用原文支持你的提取

请提供 JSON 格式的响应（必须是有效的 JSON，不要包含任何其他文本）:
{{
    "value": "提取的信息（如果未找到则为 null）",
    "confidence": 0.0-1.0,
    "reasoning": "简要解释并注明来源（例如：'来源 2 指出...'）",
    "source_quote": "支持此提取的原文引用"
}}

如果在段落中未找到信息，设置 confidence 为 0.0，value 为 null。
如果信息与核心实体无关，设置 confidence 为 0.0，value 为 null。
"""
        else:
            # English prompt
            prompt = f"""Extract the following information from the provided text passages:

QUESTION: {task.fact}
VARIABLE: {task.variable_name}
CATEGORY: {task.category}
CORE ENTITIES: {entities_str}

TEXT PASSAGES:
{combined_text}

**CRITICAL REQUIREMENTS**:
1. Extract information ONLY from the provided text passages above
2. DO NOT use your own knowledge or make assumptions
3. The extracted information MUST mention or relate to the CORE ENTITIES listed above
4. If multiple passages provide different information, prioritize the most specific and recent one
5. Quote the original text when possible to support your extraction

Please provide a JSON response (must be valid JSON, no other text):
{{
    "value": "extracted information (or null if not found)",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation with source reference (e.g., 'Source 2 states...')",
    "source_quote": "direct quote from the passage supporting this extraction"
}}

If the information is not found in the passages, set confidence to 0.0 and value to null.
If the information does not relate to the CORE ENTITIES, set confidence to 0.0 and value to null.
"""
        
        # Build system message based on language
        if lang == "zh":
            system_msg = "你是一个专业的信息提取专家。你必须只返回有效的 JSON 格式，不要包含任何其他文本。从文本段落中提取准确的事实信息。"
        else:
            system_msg = "You are an expert information extractor. You MUST respond with valid JSON only. Do not include any text before or after the JSON. Extract precise, factual information from text passages."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500,  # Increased from 500 to 1500 to avoid truncation
                response_format={"type": "json_object"}  # Force JSON response (OpenAI API)
            )
        except Exception as api_error:
            # If response_format not supported, try without it
            print(f"[EXTRACTOR][WARN] response_format not supported, retrying without it: {api_error}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500  # Increased from 500 to 1500 to avoid truncation
            )
        
        # Check for None response
        result_text = response.choices[0].message.content
        if result_text is None:
            print(f"[EXTRACTOR][ERROR] LLM returned None for {task.variable_name}")
            print(f"[EXTRACTOR][ERROR] Response finish_reason: {response.choices[0].finish_reason}")
            if response.choices[0].finish_reason == "length":
                print(f"[EXTRACTOR][ERROR] Response was truncated due to max_tokens limit!")
                print(f"[EXTRACTOR][ERROR] Consider increasing max_tokens in the code")
            return self._fallback_extract(task, passages)
        
        result_text = result_text.strip()
        
        try:
            result_data = json.loads(result_text)
            
            # Log extraction details for debugging
            if cfg.is_decision_logging_enabled('ir_rag'):
                print(f"[EXTRACTOR][DEBUG] Extracted value: {result_data.get('value')}")
                print(f"[EXTRACTOR][DEBUG] Confidence: {result_data.get('confidence')}")
                print(f"[EXTRACTOR][DEBUG] Reasoning: {result_data.get('reasoning')}")
                if result_data.get('source_quote'):
                    print(f"[EXTRACTOR][DEBUG] Source quote: {result_data.get('source_quote')[:100]}...")
            
            return ExtractedVariable(
                variable_name=task.variable_name,
                value=result_data.get("value"),
                confidence=float(result_data.get("confidence", 0.0)),
                provenance=[p.source_url for p in passages[:max_passages]],
                extraction_method="llm",
                raw_passages=[p.text for p in passages[:max_passages]]
            )
        except json.JSONDecodeError as e:
            # JSON parsing failed - log details for debugging
            print(f"[EXTRACTOR][ERROR] JSON parse failed for {task.variable_name}: {e}")
            print(f"[EXTRACTOR][ERROR] Raw response (first 300 chars): {result_text[:300]}")
            print(f"[EXTRACTOR][ERROR] This usually means LLM returned non-JSON format")
            print(f"[EXTRACTOR][ERROR] Language detected: {lang}")
            
            # Try to extract information from the text response anyway
            # Use fallback extraction as it's more reliable than raw text
            print(f"[EXTRACTOR][FALLBACK] Using fallback extraction for {task.variable_name}")
            return self._fallback_extract(task, passages)
    
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
        
        # Initialize search tool(s) based on provider configuration
        # Support multiple providers for hybrid search
        if self.config.search_providers:
            # Multiple providers configured
            self.search_tools = []
            provider_names = []
            for provider in self.config.search_providers:
                if provider == RetrievalProvider.GOOGLE_CUSTOM_SEARCH:
                    self.search_tools.append(GoogleCustomSearch())
                    provider_names.append("Google Custom Search")
                elif provider == RetrievalProvider.GCP_VERTEX_SEARCH:
                    self.search_tools.append(GCPVertexSearch())
                    provider_names.append("GCP Vertex AI Search")
                elif provider == RetrievalProvider.SERPAPI:
                    self.search_tools.append(SerpAPISearch())
                    provider_names.append("SerpAPI")
            
            # Use first tool as primary (for backward compatibility)
            self.search_tool = self.search_tools[0] if self.search_tools else SerpAPISearch()
            
            if cfg.is_decision_logging_enabled('ir_rag'):
                print(f"[IR_RAG][INIT] Using multiple search providers: {', '.join(provider_names)}")
        else:
            # Single provider configured (backward compatibility)
            self.search_tools = []
            if self.config.search_provider == RetrievalProvider.GOOGLE_CUSTOM_SEARCH:
                self.search_tool = GoogleCustomSearch()
                if cfg.is_decision_logging_enabled('ir_rag'):
                    print(f"[IR_RAG][INIT] Using Google Custom Search API")
            elif self.config.search_provider == RetrievalProvider.GCP_VERTEX_SEARCH:
                self.search_tool = GCPVertexSearch()
                if cfg.is_decision_logging_enabled('ir_rag'):
                    print(f"[IR_RAG][INIT] Using GCP Vertex AI Search")
            else:
                self.search_tool = SerpAPISearch()
                if cfg.is_decision_logging_enabled('ir_rag'):
                    print(f"[IR_RAG][INIT] Using SerpAPI")
        
        self.visitor = WebVisitor(self.config)
        self.ranker = ContentRanker(self.config)
        self.extractor = InformationExtractor(llm_client, extractor_model)
        
        if cfg.is_decision_logging_enabled('ir_rag'):
            print(f"[IR_RAG][INIT] provider={self.config.search_provider.value} max_pages={self.config.max_pages_to_visit} web_scraping={'ON' if self.config.enable_web_scraping else 'OFF'}")
    
    def _load_config_from_yaml(self) -> IRConfig:
        """Load IR_RAG configuration from YAML config file"""
        cfg = get_config()
        ir_config = cfg.get_section('ir_rag')
        
        # Parse provider(s) - support both single and multiple providers
        provider_str = ir_config.get('search', {}).get('provider', 'serpapi')
        search_providers = None
        search_provider = RetrievalProvider.SERPAPI
        
        if ',' in provider_str:
            # Multiple providers specified (e.g., "serpapi,gcp_vertex_search")
            provider_list = [p.strip() for p in provider_str.split(',')]
            search_providers = []
            for p in provider_list:
                try:
                    search_providers.append(RetrievalProvider(p))
                except ValueError:
                    print(f"[IR_RAG][WARN] Unknown provider '{p}', skipping")
            # Use first provider as primary
            if search_providers:
                search_provider = search_providers[0]
        else:
            # Single provider
            try:
                search_provider = RetrievalProvider(provider_str)
            except ValueError:
                print(f"[IR_RAG][WARN] Unknown provider '{provider_str}', using serpapi")
                search_provider = RetrievalProvider.SERPAPI
        
        return IRConfig(
            search_provider=search_provider,
            search_providers=search_providers,
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
        
        if self.config.search_provider in (RetrievalProvider.SERPAPI, RetrievalProvider.GOOGLE_CUSTOM_SEARCH, RetrievalProvider.GCP_VERTEX_SEARCH):
            return await self._web_search(query, plan, ctx)
        elif self.config.search_provider == RetrievalProvider.VECTOR_DB:
            return await self._vector_search(query, plan)
        elif self.config.search_provider == RetrievalProvider.HYBRID:
            return await self._hybrid_search(query, plan, ctx)
        else:
            raise ValueError(f"Unsupported search provider: {self.config.search_provider}")
    
    async def _web_search(self, query: str, plan: ExtractionPlan, ctx: Context) -> List[SearchResult]:
        """Search using web search API (SerpAPI, Google Custom Search, or GCP Vertex Search) with multiple diverse queries (batch mode)"""
        provider_map = {
            RetrievalProvider.GOOGLE_CUSTOM_SEARCH: "Google Custom Search",
            RetrievalProvider.GCP_VERTEX_SEARCH: "GCP Vertex AI Search",
            RetrievalProvider.SERPAPI: "SerpAPI"
        }
        
        # Check if using multiple providers
        if self.config.search_providers and len(self.config.search_providers) > 1:
            provider_names = [provider_map.get(p, str(p)) for p in self.config.search_providers]
            print(f"🔍 IR_RAG: Searching with multiple providers: {', '.join(provider_names)}...")
            return await self._multi_provider_search(query, plan, ctx)
        else:
            provider_name = provider_map.get(self.config.search_provider, "SerpAPI")
            print(f"🔍 IR_RAG: Searching with {provider_name}...")

        # Build multiple search queries from plan
        search_queries = self._build_search_query(query, plan, ctx)
        
        print(f"🔍 IR_RAG: Generated {len(search_queries)} search queries:")
        for i, q in enumerate(search_queries, 1):
            print(f"  {i}. {q}")
        print(f"🔍 IR_RAG: Time intent: {ctx.time_context.intent if ctx.time_context else 'None'}")
        
        # Use batch_call_flat to search all queries concurrently
        try:
            print(f"🔍 IR_RAG: Executing batch search with {len(search_queries)} queries...")
            
            # Get config values for batch search
            cfg = get_config()
            concurrent = cfg.get('ir_rag.search.concurrent', 8)
            qps = cfg.get('ir_rag.search.qps', 5)
            retries = cfg.get('ir_rag.search.retries', 2)
            
            # Execute batch search (synchronous but internally concurrent)
            raw_results = self.search_tool.batch_call_flat(
                queries=search_queries,
                num_results=self.config.max_search_results,
                location=self.config.search_location,
                language=self.config.search_language,
                concurrent=concurrent,
                qps=qps,
                retries=retries,
                tbs="qdr:y"
            )
            
            print(f"🔍 IR_RAG: Batch search returned {len(raw_results)} total results")
            
            # Convert raw results to SearchResult objects and deduplicate by URL
            all_search_results = []
            seen_urls = set()
            
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
            
            print(f"🔍 IR_RAG: Retrieved {len(all_search_results)} unique results from {len(search_queries)} queries")
            return all_search_results
            
        except Exception as e:
            logger.error(f"[IR_RAG][SEARCH] Batch search failed: {e}")
            return []
    
    async def _multi_provider_search(self, query: str, plan: ExtractionPlan, ctx: Context) -> List[SearchResult]:
        """Search using multiple providers and merge results"""
        print(f"🔍 IR_RAG: Executing multi-provider search with {len(self.search_tools)} providers...")
        
        # Build search queries
        search_queries = self._build_search_query(query, plan, ctx)
        print(f"🔍 IR_RAG: Generated {len(search_queries)} search queries for each provider")
        
        # Get config values for batch search
        cfg = get_config()
        concurrent = cfg.get('ir_rag.search.concurrent', 8)
        qps = cfg.get('ir_rag.search.qps', 5)
        retries = cfg.get('ir_rag.search.retries', 2)
        
        all_results = []
        provider_stats = {}
        
        # Execute search with each provider
        for i, search_tool in enumerate(self.search_tools):
            provider_name = self.config.search_providers[i].value if i < len(self.config.search_providers) else f"Provider {i+1}"
            
            try:
                print(f"🔍 IR_RAG: Searching with {provider_name}...")
                
                # Execute batch search
                raw_results = search_tool.batch_call_flat(
                    queries=search_queries,
                    num_results=self.config.max_search_results,
                    location=self.config.search_location,
                    language=self.config.search_language,
                    concurrent=concurrent,
                    qps=qps,
                    retries=retries,
                    tbs="qdr:y"
                )
                
                provider_stats[provider_name] = len(raw_results)
                print(f"🔍 IR_RAG: {provider_name} returned {len(raw_results)} results")
                
                # Convert to SearchResult objects
                for result in raw_results:
                    search_result = SearchResult(
                        title=result.get('title', ''),
                        snippet=result.get('snippet', ''),
                        url=result.get('link', ''),
                        source=result.get('source', ''),
                        result_type=result.get('type', 'organic'),
                        position=result.get('position', 0),
                        date=result.get('date'),
                        confidence=0.8
                    )
                    all_results.append(search_result)
                    
            except Exception as e:
                logger.error(f"[IR_RAG][SEARCH] {provider_name} failed: {e}")
                provider_stats[provider_name] = 0
                # Continue with other providers even if one fails
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        print(f"🔍 IR_RAG: Multi-provider search stats: {provider_stats}")
        print(f"🔍 IR_RAG: Total results: {len(all_results)}, Unique results: {len(unique_results)}")
        
        return unique_results
    
    async def _vector_search(self, query: str, plan: ExtractionPlan) -> List[SearchResult]:
        """Search using vector database (placeholder)"""
        print(f"🔍 IR_RAG: Vector search not implemented yet, using fallback...")
        
        # Placeholder for future vector database integration
        # This would query a vector database with embeddings
        
        return []
    
    async def _hybrid_search(self, query: str, plan: ExtractionPlan, ctx: Context) -> List[SearchResult]:
        """Combine web search and vector search results"""
        print(f"🔍 IR_RAG: Hybrid search not fully implemented, using web search only...")
        
        # For now, just use web search
        # In the future, this would combine results from both sources
        return await self._web_search(query, plan, ctx)
    
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
        
        # Get current time information
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_date = datetime.datetime.now().strftime("%Y年%m月%d日")
        
        # Build time context string
        time_info = f"Current time: {current_time} ({current_date})"
        if ctx.time_context:
            time_info += f"\nTime intent: {ctx.time_context.intent}"
            if ctx.time_context.window and ctx.time_context.window[1]:
                time_info += f"\nTime window: {ctx.time_context.window[0]} to {ctx.time_context.window[1]}"
        
        # Create synthesis prompt
        synthesis_prompt = f"""
        Based on the extracted information, provide a comprehensive answer to: {ctx.query}. You MUST use all retrieved information, no missing information. Be as comprehensive as possible. Add background information if there is any possible.
        
        **IMPORTANT TIME CONTEXT**:
        {time_info}
        
        Use the following extracted information:
        {materials}
        
        Please provide a well-structured response with:
        1. Clear sections based on the information found
        2. DO NOT include any citation numbers like [1], [2], etc. in the main text
        3. Use diverse formatting: include bullet points, numbered lists, comparison tables where appropriate
        4. For key facts/statistics/comparisons, use tables or lists instead of long paragraphs
        5. Mix paragraph text with structured formats (lists, tables) for better readability
        6. Factual accuracy based on the sources
        7. **Time awareness**: The current time is {current_time}. Filter and present information based on this current time. Distinguish between past, present, and future events clearly.
        8. Include a "References" or "参考来源" section at the very end listing all sources
        9. MUST use all retrieved information, no missing information.
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
                "instruction_hint": f"Provide a comprehensive answer with clear sections. Use diverse formatting: bullet points, numbered lists, and tables where appropriate. Mix paragraphs with structured formats for better readability. DO NOT use citation numbers [1], [2], etc. in the main text. Only include a reference list at the end.\n\n**CRITICAL**: The current time is {current_time} ({current_date}). Be aware of this when discussing events, dates, and timelines. Clearly distinguish between past, present, and future."
            }
            
            response = await toolset.synthesizer.generate(
                category="INFORMATION_RETRIEVAL",
                style_key="auto",  # 使用 auto 启用自动语言检测和样式选择
                constraints=constraints,
                materials=f"Query: {ctx.query}\n\n**Current Time**: {current_time} ({current_date})\n\nExtracted Information:\n{materials}",
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
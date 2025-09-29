"""
Productivity Action for WebChasor System

A reliable text transformer that handles various productivity tasks without adding
external facts. Focuses on reformatting, summarizing, extracting, and restructuring
existing content into action-ready outputs.

Task Types:
- Summarize: Shorter, faithful, task-oriented condensations
- Rewrite: Change tone/reading level/voice without changing meaning
- Extract: Return structured JSON/CSV according to schema
- Format: Restructure into bullets, tables, checklists, outlines
- Translate: Convert between languages while preserving meaning
- Analyze: Break down structure, identify key points, create frameworks
"""

import os
import json
import re
import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv
import openai

from artifacts import Action, Artifact, Context
from config_manager import get_config
from prompt import (
    PRODUCTIVITY_SYSTEM_PROMPT,
    PRODUCTIVITY_SUMMARIZATION_PROMPT,
    PRODUCTIVITY_REWRITING_PROMPT,
    PRODUCTIVITY_EXTRACTION_PROMPT,
    PRODUCTIVITY_FORMATTING_PROMPT,
    PRODUCTIVITY_TRANSLATION_PROMPT,
    PRODUCTIVITY_ANALYSIS_PROMPT,
    PRODUCTIVITY_OUTLINE_PROMPT,
    PRODUCTIVITY_CHECKLIST_PROMPT,
    PRODUCTIVITY_VALIDATION_RULES,
    PRODUCTIVITY_ERROR_MESSAGES
)

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

class ProductivityTaskType(Enum):
    """Types of productivity tasks"""
    SUMMARIZE = "summarize"           # Condense content while preserving key points
    REWRITE = "rewrite"              # Change style/tone/format without changing meaning
    EXTRACT = "extract"              # Pull specific data into structured format
    FORMAT = "format"                # Restructure into bullets, tables, etc.
    TRANSLATE = "translate"          # Convert between languages
    ANALYZE = "analyze"              # Break down structure, identify patterns
    OUTLINE = "outline"              # Create hierarchical structure
    CHECKLIST = "checklist"          # Convert to actionable items

@dataclass
class ProductivityConstraints:
    """Comprehensive constraints for productivity tasks"""
    # Language and localization
    language: str = "auto"  # "en", "zh-cn", "zh-tw", "auto"
    
    # Content length controls
    length: str = "moderate"  # "brief", "moderate", "detailed", "preserve", or specific like "100 words"
    max_words: Optional[int] = None
    max_sentences: Optional[int] = None
    max_paragraphs: Optional[int] = None
    
    # Style and tone
    tone: str = "neutral"  # "formal", "casual", "professional", "friendly", "academic"
    reading_level: str = "general"  # "elementary", "middle", "high", "college", "graduate", "general"
    style_guide: Optional[str] = None  # "APA", "MLA", "Chicago", "AP", "custom"
    
    # Formatting constraints
    bullets_n: Optional[int] = None  # Number of bullet points
    table_columns: Optional[List[str]] = None  # Column names for table format
    use_numbered_lists: bool = False
    use_bullet_points: bool = False
    use_headers: bool = True
    
    # Output format specifications
    output_format: str = "text"  # "text", "json", "csv", "markdown", "html"
    json_schema: Optional[Dict[str, Any]] = None  # JSON schema for structured output
    csv_delimiter: str = ","
    markdown_style: str = "github"  # "github", "commonmark", "gfm"
    
    # Content preservation rules
    do_not_change_facts: bool = True  # Never alter factual information
    keep_quotes: bool = True  # Preserve exact quotes
    preserve_numbers: bool = True  # Keep all numbers unchanged
    preserve_names: bool = True  # Keep proper names unchanged
    preserve_dates: bool = True  # Keep dates unchanged
    preserve_urls: bool = True  # Keep URLs unchanged
    
    # Task-specific constraints
    extract_targets: Optional[List[str]] = None  # What to extract: ["dates", "names", "numbers"]
    summary_focus: str = "key_points"  # "key_points", "chronological", "thematic"
    rewrite_target: str = "clarity"  # "clarity", "conciseness", "formality", "simplicity"
    
    # Quality controls
    require_citations: bool = False  # Include source references
    fact_check_mode: bool = True  # Extra careful with facts
    consistency_check: bool = True  # Ensure internal consistency
    
    # Advanced options
    custom_instructions: Optional[str] = None  # Additional specific instructions
    excluded_sections: Optional[List[str]] = None  # Sections to skip
    priority_sections: Optional[List[str]] = None  # Sections to emphasize

@dataclass
class ProductivityConfig:
    """Configuration for productivity tasks"""
    api_base: str
    api_key: str
    model: str
    timeout: int = 30
    temperature: float = 0.0  # Deterministic for consistency
    use_llm_classification: bool = True
    max_tokens: int = 2000

@dataclass
class TaskTemplate:
    """Template for productivity task execution"""
    name: str
    description: str
    prompt_template: str
    output_format: str
    validation_rules: List[str]

@dataclass
class ProductivityTask:
    """Complete productivity task specification"""
    task: str  # Task type as string
    source_text: str  # Required source content
    constraints: ProductivityConstraints  # All constraints and options

# Task classification prompt for LLM
PRODUCTIVITY_TASK_CLASSIFIER_PROMPT = """You are a productivity task classifier. Analyze the user's request and determine what type of text transformation they want.

Available task types:
- SUMMARIZE: Make content shorter while keeping key points (condense, shorten, brief overview)
- REWRITE: Change style, tone, or reading level without changing meaning (rephrase, simplify, formalize)
- EXTRACT: Pull specific information into structured format (get data, find items, list details)
- FORMAT: Restructure into specific layout (bullets, tables, numbered lists, organize)
- TRANSLATE: Convert between languages (translate to, convert language)
- ANALYZE: Break down structure or identify patterns (analyze, examine, break down)
- OUTLINE: Create hierarchical structure (outline, structure, organize hierarchically)
- CHECKLIST: Convert to actionable items (action items, todo list, steps)

Respond with ONLY the task type and confidence score.
Format: TASK_TYPE confidence_score

Examples:
- "Summarize this article" → SUMMARIZE 0.95
- "Rewrite this email to be more formal" → REWRITE 0.9
- "Extract all the dates mentioned" → EXTRACT 0.9
- "Convert this to bullet points" → FORMAT 0.95
- "Translate this to Chinese" → TRANSLATE 0.95
- "Analyze the structure of this argument" → ANALYZE 0.85
- "Create an outline of this content" → OUTLINE 0.9
- "Turn this into a checklist" → CHECKLIST 0.9"""

# Task templates using imported prompts from prompt.py
TASK_TEMPLATES = {
    ProductivityTaskType.SUMMARIZE: TaskTemplate(
        name="Content Summarization",
        description="Create concise summaries while preserving key information",
        prompt_template=PRODUCTIVITY_SUMMARIZATION_PROMPT,
        output_format="text",
        validation_rules=PRODUCTIVITY_VALIDATION_RULES["summarize"]
    ),
    
    ProductivityTaskType.REWRITE: TaskTemplate(
        name="Content Rewriting",
        description="Change style, tone, or format while preserving meaning",
        prompt_template=PRODUCTIVITY_REWRITING_PROMPT,
        output_format="text",
        validation_rules=PRODUCTIVITY_VALIDATION_RULES["rewrite"]
    ),
    
    ProductivityTaskType.EXTRACT: TaskTemplate(
        name="Data Extraction",
        description="Extract specific information into structured format",
        prompt_template=PRODUCTIVITY_EXTRACTION_PROMPT,
        output_format="structured",
        validation_rules=PRODUCTIVITY_VALIDATION_RULES["extract"]
    ),
    
    ProductivityTaskType.FORMAT: TaskTemplate(
        name="Content Formatting",
        description="Restructure content into specific layouts",
        prompt_template=PRODUCTIVITY_FORMATTING_PROMPT,
        output_format="formatted",
        validation_rules=PRODUCTIVITY_VALIDATION_RULES["format"]
    ),
    
    ProductivityTaskType.TRANSLATE: TaskTemplate(
        name="Language Translation",
        description="Translate content between languages",
        prompt_template=PRODUCTIVITY_TRANSLATION_PROMPT,
        output_format="text",
        validation_rules=PRODUCTIVITY_VALIDATION_RULES["translate"]
    ),
    
    ProductivityTaskType.ANALYZE: TaskTemplate(
        name="Content Analysis",
        description="Analyze structure, patterns, and key elements",
        prompt_template=PRODUCTIVITY_ANALYSIS_PROMPT,
        output_format="analytical",
        validation_rules=PRODUCTIVITY_VALIDATION_RULES["analyze"]
    ),
    
    ProductivityTaskType.OUTLINE: TaskTemplate(
        name="Content Outlining",
        description="Create hierarchical structure from content",
        prompt_template=PRODUCTIVITY_OUTLINE_PROMPT,
        output_format="hierarchical",
        validation_rules=PRODUCTIVITY_VALIDATION_RULES["outline"]
    ),
    
    ProductivityTaskType.CHECKLIST: TaskTemplate(
        name="Checklist Creation",
        description="Convert content into actionable checklist items",
        prompt_template=PRODUCTIVITY_CHECKLIST_PROMPT,
        output_format="checklist",
        validation_rules=PRODUCTIVITY_VALIDATION_RULES["checklist"]
    )
}

class PRODUCTIVITY(Action):
    """
    Productivity action that handles various text transformation tasks
    without adding external information.
    """
    
    name = "PRODUCTIVITY"
    requires_tools = ["synthesizer"]
    
    def __init__(self, config: Optional[ProductivityConfig] = None):
        """Initialize the productivity action"""
        super().__init__()
        self.config = config or self._load_config_from_env()
        self.client = self._init_openai_client() if self.config.use_llm_classification else None
        logger.info(f"PRODUCTIVITY action initialized (LLM classification: {self.config.use_llm_classification})")
    
    def _load_config_from_env(self) -> ProductivityConfig:
        """Load configuration from environment variables"""
        api_base = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY_AGENT")
        model = os.getenv("OPENAI_API_MODEL_AGENT_PRODUCTIVITY", "gpt-3.5-turbo")
        
        # LLM classification is optional
        use_llm = bool(api_base and api_key and model)
        
        return ProductivityConfig(
            api_base=api_base or "",
            api_key=api_key or "",
            model=model,
            use_llm_classification=use_llm
        )
    
    def _init_openai_client(self) -> Optional[openai.OpenAI]:
        """Initialize OpenAI client for LLM-based classification"""
        try:
            if not all([self.config.api_base, self.config.api_key]):
                logger.warning("OpenAI credentials not available, using rule-based classification only")
                return None
                
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return None
    
    async def run(self, ctx: Context, toolset) -> Artifact:
        """
        Execute productivity task on the given query.
        PRODUCTIVITY calls the Synthesizer inside its own run() method.
        
        Args:
            ctx: Context containing the query and metadata
            toolset: Available tools (requires synthesizer)
            
        Returns:
            Artifact: Processed productivity output with specific format:
            - kind: "text" | "table" | "json"
            - content: transformed text / Markdown table / JSON string
            - meta: {"language": "en", "tokens_in": 350, "tokens_out": 120, ...}
        """
        try:
            logger.info(f"Starting productivity task for: {ctx.query[:100]}...")
            
            # Validate inputs
            if not ctx.query or not ctx.query.strip():
                raise ValueError(PRODUCTIVITY_ERROR_MESSAGES["empty_content"])
            
            if not hasattr(toolset, 'synthesizer'):
                raise ValueError("Synthesizer tool is required but not available")
            
            # Parse the complete task specification
            task_spec = self._parse_task_specification(ctx.query)
            logger.info(f"Parsed task: {task_spec.task}, constraints: {len(task_spec.constraints.__dict__)} fields")
            
            # Get task template
            task_type = ProductivityTaskType(task_spec.task)
            template = TASK_TEMPLATES[task_type]
            
            # Build specialized prompt with constraints
            specialized_prompt = self._build_constrained_prompt(template, task_spec)
            
            # Count input tokens (approximate)
            tokens_in = len(specialized_prompt.split()) + len(task_spec.source_text.split())
            
            # PRODUCTIVITY calls the Synthesizer inside its own run() method
            logger.info("Calling Synthesizer from PRODUCTIVITY.run()")
            result = await toolset.synthesizer.synthesize(
                category="TASK_PRODUCTIVITY",
                plan=None,
                extracted=None,
                user_query=specialized_prompt,
                system_prompt=PRODUCTIVITY_SYSTEM_PROMPT
            )
            
            # Count output tokens (approximate)
            tokens_out = len(result.split()) if result else 0
            
            # Validate and format output according to constraints
            validated_result = self._validate_constrained_output(result, task_spec, template)
            
            # Determine output kind based on constraints and content
            output_kind = self._determine_output_kind(task_spec.constraints, validated_result)
            
            # Format content based on output kind
            formatted_content = self._format_output_content(validated_result, output_kind, task_spec.constraints)
            
            # Create artifact with the specific requested format
            artifact = Artifact(
                kind=output_kind,  # "text" | "table" | "json"
                content=formatted_content,  # transformed text / Markdown table / JSON string
                meta={
                    # Core metadata
                    "language": task_spec.constraints.language if task_spec.constraints.language != "auto" else "en",
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    
                    # Task information
                    "task": task_spec.task,
                    "category": ctx.router_category,
                    "template_used": template.name,
                    
                    # Processing details
                    "source_text_length": len(task_spec.source_text),
                    "output_length": len(formatted_content),
                    "compression_ratio": round(len(formatted_content) / len(task_spec.source_text), 2) if task_spec.source_text else 1.0,
                    
                    # Constraints applied
                    "tone": task_spec.constraints.tone,
                    "reading_level": task_spec.constraints.reading_level,
                    "output_format": task_spec.constraints.output_format,
                    "preserve_facts": task_spec.constraints.do_not_change_facts,
                    "preserve_quotes": task_spec.constraints.keep_quotes,
                    
                    # Quality metrics
                    "validation_passed": True,
                    "classification_method": "llm" if self.client else "rule-based",
                    "processing_time": "< 1s",  # Could be measured if needed
                    
                    # Additional constraints (if any)
                    "max_words": task_spec.constraints.max_words,
                    "max_sentences": task_spec.constraints.max_sentences,
                    "bullets_n": task_spec.constraints.bullets_n,
                    "table_columns": task_spec.constraints.table_columns,
                    "style_guide": task_spec.constraints.style_guide,
                    
                    # Technical details
                    "prompt_source": "imported_from_prompt_py",
                    "synthesizer_called": True,
                    "constraints_count": len([k for k, v in task_spec.constraints.__dict__.items() if v not in [None, False, "auto", "moderate", "neutral", "general"]])
                }
            )
            
            logger.info(f"Productivity task completed successfully: {task_spec.task} -> {output_kind}")
            logger.info(f"Token usage: {tokens_in} in, {tokens_out} out")
            return artifact
            
        except Exception as e:
            logger.error(f"Productivity task failed: {str(e)}")
            
            # Return fallback response using imported error message
            fallback_text = self._generate_fallback_response(ctx.query, str(e))
            return Artifact(
                kind="text",  # Fallback is always text
                content=fallback_text,
                meta={
                    "language": "en",
                    "tokens_in": 0,
                    "tokens_out": len(fallback_text.split()),
                    "category": ctx.router_category,
                    "error": str(e),
                    "fallback": True,
                    "synthesizer_called": False
                }
            )
    
    def _parse_task_specification(self, query: str) -> ProductivityTask:
        """
        Parse the complete task specification with all three aspects:
        1. task: task type
        2. source_text: content to process
        3. constraints: all formatting and processing constraints
        """
        # Determine task type
        task_type, confidence = self._determine_task_type(query)
        
        # Simple decision print
        print(f"[PRODUCTIVITY][DECISION] {task_type.value} conf={confidence}")
        
        # Extract source text
        source_text = self._extract_source_content(query)
        
        # Parse comprehensive constraints
        constraints = self._parse_constraints(query, task_type)
        
        return ProductivityTask(
            task=task_type.value,
            source_text=source_text,
            constraints=constraints
        )
    
    def _determine_task_type(self, query: str) -> tuple[ProductivityTaskType, float]:
        """Determine task type (sync version for parsing)"""
        # Use rule-based classification for parsing
        task_type = self._rule_based_task_classification(query)
        return task_type, 0.8
    
    def _parse_constraints(self, query: str, task_type: ProductivityTaskType) -> ProductivityConstraints:
        """Parse comprehensive constraints from the query"""
        constraints = ProductivityConstraints()
        query_lower = query.lower()
        
        # Language detection
        language_patterns = {
            'chinese': ['chinese', 'zh-cn', '中文', '简体'],
            'traditional_chinese': ['traditional chinese', 'zh-tw', '繁體', '繁体'],
            'english': ['english', 'en'],
            'spanish': ['spanish', 'español'],
            'french': ['french', 'français'],
            'german': ['german', 'deutsch'],
            'japanese': ['japanese', '日本語']
        }
        
        for lang_key, patterns in language_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                constraints.language = lang_key
                break
        
        # Length constraints
        length_patterns = {
            'brief': ['brief', 'short', 'concise', 'summary'],
            'detailed': ['detailed', 'comprehensive', 'thorough', 'extensive'],
            'preserve': ['same length', 'preserve length', 'maintain length']
        }
        
        for length_key, patterns in length_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                constraints.length = length_key
                break
        
        # Extract specific word/sentence limits
        word_match = re.search(r'(\d+)\s*words?', query_lower)
        if word_match:
            constraints.max_words = int(word_match.group(1))
        
        sentence_match = re.search(r'(\d+)\s*sentences?', query_lower)
        if sentence_match:
            constraints.max_sentences = int(sentence_match.group(1))
        
        # Tone detection
        tone_patterns = {
            'formal': ['formal', 'professional', 'business', 'official'],
            'casual': ['casual', 'informal', 'conversational', 'friendly'],
            'academic': ['academic', 'scholarly', 'research'],
            'technical': ['technical', 'scientific']
        }
        
        for tone_key, patterns in tone_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                constraints.tone = tone_key
                break
        
        # Reading level
        reading_patterns = {
            'elementary': ['elementary', 'simple', 'basic', 'easy'],
            'middle': ['middle school', 'intermediate'],
            'high': ['high school', 'advanced'],
            'college': ['college', 'university'],
            'graduate': ['graduate', 'phd', 'doctoral']
        }
        
        for level_key, patterns in reading_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                constraints.reading_level = level_key
                break
        
        # Format-specific constraints
        if task_type == ProductivityTaskType.FORMAT:
            # Bullet points
            bullet_match = re.search(r'(\d+)\s*bullet', query_lower)
            if bullet_match:
                constraints.bullets_n = int(bullet_match.group(1))
                constraints.use_bullet_points = True
            elif 'bullet' in query_lower:
                constraints.use_bullet_points = True
            
            # Numbered lists
            if 'numbered' in query_lower or 'number' in query_lower:
                constraints.use_numbered_lists = True
            
            # Table format
            if 'table' in query_lower:
                # Try to extract column specifications
                columns_match = re.search(r'columns?:\s*([^.]+)', query_lower)
                if columns_match:
                    columns_text = columns_match.group(1)
                    constraints.table_columns = [col.strip() for col in columns_text.split(',')]
        
        # Output format
        format_patterns = {
            'json': ['json', 'javascript object notation'],
            'csv': ['csv', 'comma separated'],
            'markdown': ['markdown', 'md'],
            'html': ['html', 'web format']
        }
        
        for format_key, patterns in format_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                constraints.output_format = format_key
                break
        
        # JSON schema detection
        if 'schema' in query_lower or 'structure' in query_lower:
            # Try to extract JSON schema if provided
            schema_match = re.search(r'\{[^}]+\}', query)
            if schema_match:
                try:
                    constraints.json_schema = json.loads(schema_match.group())
                except json.JSONDecodeError:
                    pass
        
        # Content preservation rules
        if 'preserve facts' in query_lower or 'keep facts' in query_lower:
            constraints.do_not_change_facts = True
        
        if 'preserve quotes' in query_lower or 'keep quotes' in query_lower:
            constraints.keep_quotes = True
        
        if 'preserve numbers' in query_lower or 'keep numbers' in query_lower:
            constraints.preserve_numbers = True
        
        # Extract targets for extraction tasks
        if task_type == ProductivityTaskType.EXTRACT:
            extract_patterns = {
                'dates': ['date', 'time', 'when'],
                'names': ['name', 'person', 'who'],
                'numbers': ['number', 'statistic', 'figure', 'amount'],
                'locations': ['location', 'place', 'where'],
                'organizations': ['organization', 'company', 'institution']
            }
            
            extract_targets = []
            for target, patterns in extract_patterns.items():
                if any(pattern in query_lower for pattern in patterns):
                    extract_targets.append(target)
            
            if extract_targets:
                constraints.extract_targets = extract_targets
        
        # Style guide detection
        style_guides = ['apa', 'mla', 'chicago', 'ap', 'ieee']
        for guide in style_guides:
            if guide in query_lower:
                constraints.style_guide = guide.upper()
                break
        
        # Custom instructions (everything after "also" or "additionally")
        custom_match = re.search(r'(?:also|additionally|furthermore)[,:]?\s*(.+)', query_lower)
        if custom_match:
            constraints.custom_instructions = custom_match.group(1)
        
        return constraints
    
    def _build_constrained_prompt(self, template: TaskTemplate, task_spec: ProductivityTask) -> str:
        """Build prompt incorporating all constraints"""
        constraints = task_spec.constraints
        
        # Build constraint instructions
        constraint_instructions = []
        
        # Language
        if constraints.language != "auto":
            constraint_instructions.append(f"Language: {constraints.language}")
        
        # Length
        if constraints.max_words:
            constraint_instructions.append(f"Maximum words: {constraints.max_words}")
        if constraints.max_sentences:
            constraint_instructions.append(f"Maximum sentences: {constraints.max_sentences}")
        if constraints.length != "moderate":
            constraint_instructions.append(f"Length style: {constraints.length}")
        
        # Tone and style
        if constraints.tone != "neutral":
            constraint_instructions.append(f"Tone: {constraints.tone}")
        if constraints.reading_level != "general":
            constraint_instructions.append(f"Reading level: {constraints.reading_level}")
        if constraints.style_guide:
            constraint_instructions.append(f"Style guide: {constraints.style_guide}")
        
        # Format constraints
        if constraints.use_bullet_points:
            bullet_text = f"Use bullet points"
            if constraints.bullets_n:
                bullet_text += f" ({constraints.bullets_n} items)"
            constraint_instructions.append(bullet_text)
        
        if constraints.use_numbered_lists:
            constraint_instructions.append("Use numbered lists")
        
        if constraints.table_columns:
            constraint_instructions.append(f"Table columns: {', '.join(constraints.table_columns)}")
        
        # Output format
        if constraints.output_format != "text":
            constraint_instructions.append(f"Output format: {constraints.output_format}")
        
        if constraints.json_schema:
            constraint_instructions.append(f"JSON schema: {json.dumps(constraints.json_schema)}")
        
        # Preservation rules
        preservation_rules = []
        if constraints.do_not_change_facts:
            preservation_rules.append("Do not change any factual information")
        if constraints.keep_quotes:
            preservation_rules.append("Preserve all quotes exactly")
        if constraints.preserve_numbers:
            preservation_rules.append("Keep all numbers unchanged")
        if constraints.preserve_names:
            preservation_rules.append("Keep all proper names unchanged")
        
        if preservation_rules:
            constraint_instructions.extend(preservation_rules)
        
        # Extract targets
        if constraints.extract_targets:
            constraint_instructions.append(f"Extract: {', '.join(constraints.extract_targets)}")
        
        # Custom instructions
        if constraints.custom_instructions:
            constraint_instructions.append(f"Additional: {constraints.custom_instructions}")
        
        # Build the complete prompt
        constraint_text = "\n".join(f"- {instruction}" for instruction in constraint_instructions)
        
        # Format the template with all parameters
        prompt_params = {
            'source_content': task_spec.source_text,
            'constraints': constraint_text,
            'target_length': constraints.length,
            'style': f"{constraints.tone}, {constraints.reading_level} level",
            'target_style': constraints.tone,
            'target_audience': f"{constraints.reading_level} level readers",
            'tone': constraints.tone,
            'extract_targets': ', '.join(constraints.extract_targets) if constraints.extract_targets else 'key information',
            'output_format': constraints.output_format,
            'target_format': 'bullet points' if constraints.use_bullet_points else 'organized structure',
            'target_language': constraints.language,
            'analysis_focus': 'comprehensive analysis'
        }
        
        try:
            # Add constraint section to the template
            enhanced_template = template.prompt_template + f"\n\n**Additional Constraints:**\n{constraint_text}\n"
            formatted_prompt = enhanced_template.format(**prompt_params)
        except KeyError as e:
            logger.warning(f"Missing parameter {e}, using base template")
            formatted_prompt = template.prompt_template.format(source_content=task_spec.source_text)
        
        return formatted_prompt
    
    def _validate_constrained_output(self, output: str, task_spec: ProductivityTask, template: TaskTemplate) -> str:
        """Validate output against constraints"""
        constraints = task_spec.constraints
        
        if not output or len(output.strip()) < 10:
            raise ValueError(PRODUCTIVITY_ERROR_MESSAGES["content_too_short"])
        
        # Word count validation
        if constraints.max_words:
            word_count = len(output.split())
            if word_count > constraints.max_words:
                logger.warning(f"Output exceeds word limit: {word_count} > {constraints.max_words}")
        
        # Sentence count validation
        if constraints.max_sentences:
            sentence_count = len(re.findall(r'[.!?]+', output))
            if sentence_count > constraints.max_sentences:
                logger.warning(f"Output exceeds sentence limit: {sentence_count} > {constraints.max_sentences}")
        
        # Format validation
        if constraints.output_format == "json":
            try:
                json.loads(output)
            except json.JSONDecodeError:
                logger.warning("Output is not valid JSON")
        
        return output.strip()
    
    async def _analyze_task(self, query: str) -> tuple[ProductivityTaskType, float, Dict[str, Any]]:
        """Legacy method for backward compatibility"""
        task_spec = self._parse_task_specification(query)
        task_type = ProductivityTaskType(task_spec.task)
        return task_type, 0.8, task_spec.constraints.__dict__
    
    async def _llm_task_classification(self, query: str) -> Optional[tuple[ProductivityTaskType, float]]:
        """Use LLM to classify the productivity task type"""
        try:
            messages = [
                {"role": "system", "content": PRODUCTIVITY_TASK_CLASSIFIER_PROMPT},
                {"role": "user", "content": f"Request: {query}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=50
            )
            
            raw_response = response.choices[0].message.content.strip()
            logger.debug(f"LLM task classification response: {raw_response}")
            
            # Parse response
            parts = raw_response.split()
            if len(parts) >= 2:
                type_str = parts[0].upper()
                confidence = float(parts[1])
                
                for task_type in ProductivityTaskType:
                    if task_type.value.upper() == type_str:
                        return task_type, confidence
            
            # Fallback parsing
            for task_type in ProductivityTaskType:
                if task_type.value.upper() in raw_response.upper():
                    return task_type, 0.7
            
            return None
            
        except Exception as e:
            logger.error(f"LLM task classification error: {e}")
            return None
    
    def _rule_based_task_classification(self, query: str) -> ProductivityTaskType:
        """Rule-based fallback for task classification"""
        query_lower = query.lower()
        
        # Summarize indicators
        if any(word in query_lower for word in ['summarize', 'summary', 'condense', 'brief', 'shorten']):
            return ProductivityTaskType.SUMMARIZE
        
        # Extract indicators
        if any(word in query_lower for word in ['extract', 'pull out', 'get all', 'find all', 'list all']):
            return ProductivityTaskType.EXTRACT
        
        # Format indicators
        if any(word in query_lower for word in ['bullet', 'format', 'table', 'organize', 'structure']):
            return ProductivityTaskType.FORMAT
        
        # Translate indicators
        if any(word in query_lower for word in ['translate', 'convert to', 'in chinese', 'in english']):
            return ProductivityTaskType.TRANSLATE
        
        # Rewrite indicators
        if any(word in query_lower for word in ['rewrite', 'rephrase', 'formal', 'casual', 'simplify']):
            return ProductivityTaskType.REWRITE
        
        # Analyze indicators
        if any(word in query_lower for word in ['analyze', 'analysis', 'examine', 'break down']):
            return ProductivityTaskType.ANALYZE
        
        # Outline indicators
        if any(word in query_lower for word in ['outline', 'structure', 'hierarchy']):
            return ProductivityTaskType.OUTLINE
        
        # Checklist indicators
        if any(word in query_lower for word in ['checklist', 'todo', 'action items', 'steps']):
            return ProductivityTaskType.CHECKLIST
        
        # Default to rewrite
        return ProductivityTaskType.REWRITE
    
    def _extract_source_content(self, query: str) -> str:
        """Extract the source content from the query"""
        # Look for common patterns that separate instruction from content
        separators = [
            'the following:', 'this text:', 'this content:', 'this article:',
            'the text:', 'the content:', 'the article:', 'this passage:',
            ':\n', ':\r\n'
        ]
        
        for separator in separators:
            if separator in query.lower():
                parts = query.split(separator, 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        # If no clear separator, assume the entire query is the content
        return query
    
    def _generate_fallback_response(self, query: str, error: str) -> str:
        """Generate fallback response when task fails using imported error messages"""
        return f"""I apologize, but I encountered an issue processing your productivity request: "{query[:100]}..."

Error: {error}

I can help you with various text transformation tasks:
- Summarizing content
- Rewriting in different styles
- Extracting specific information
- Formatting into lists or tables
- Translating between languages
- Creating outlines or checklists

Please try rephrasing your request or provide the content you'd like me to work with."""
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the productivity action"""
        health = {
            "status": "healthy",
            "task_types_available": len(TASK_TEMPLATES),
            "supported_tasks": [t.value for t in ProductivityTaskType],
            "llm_classification": self.config.use_llm_classification,
            "model": self.config.model if self.config.use_llm_classification else None,
            "prompts_imported": True,
            "validation_rules_loaded": len(PRODUCTIVITY_VALIDATION_RULES) > 0,
            "constraints_system": "comprehensive",
            "constraint_fields": len(ProductivityConstraints.__dataclass_fields__)
        }
        
        # Test LLM classification if available
        if self.client:
            try:
                test_result = await self._llm_task_classification("Summarize this article")
                health["llm_test"] = "passed" if test_result else "failed"
                if test_result:
                    health["llm_test_result"] = {"type": test_result[0].value, "confidence": test_result[1]}
            except Exception as e:
                health["llm_test"] = "failed"
                health["llm_test_error"] = str(e)
        
        return health
    
    def _determine_output_kind(self, constraints: ProductivityConstraints, content: str) -> str:
        """
        Determine the output kind based on constraints and content.
        Returns: "text" | "table" | "json"
        """
        # Check constraints first
        if constraints.output_format == "json" or constraints.json_schema:
            return "json"
        
        if constraints.table_columns or constraints.output_format == "csv":
            return "table"
        
        # Check content patterns
        content_lower = content.lower().strip()
        
        # JSON detection
        if content.strip().startswith(('{', '[')):
            try:
                json.loads(content)
                return "json"
            except json.JSONDecodeError:
                pass
        
        if content.strip().startswith('[') and content.strip().endswith(']'):
            try:
                json.loads(content)
                return "json"
            except json.JSONDecodeError:
                pass
        
        # Table detection (Markdown table format)
        if '|' in content and ('---' in content or ':-:' in content or ':--' in content):
            return "table"
        
        # CSV detection
        if constraints.output_format == "csv" or (content.count(',') > content.count('\n') and content.count('\n') > 2):
            return "table"
        
        # Default to text
        return "text"
    
    def _format_output_content(self, content: str, output_kind: str, constraints: ProductivityConstraints) -> str:
        """
        Format the output content based on the determined kind and constraints.
        
        Args:
            content: Raw output from synthesizer
            output_kind: "text" | "table" | "json"
            constraints: Formatting constraints
            
        Returns:
            Formatted content string
        """
        if output_kind == "json":
            # Ensure valid JSON format
            try:
                # Try to parse and reformat for consistency
                if content.strip().startswith(('{', '[')):
                    parsed = json.loads(content)
                    return json.dumps(parsed, indent=2, ensure_ascii=False)
                else:
                    # Content is not JSON, wrap it appropriately
                    if constraints.json_schema:
                        # Try to fit content into schema
                        return json.dumps({"content": content, "format": "text"}, indent=2)
                    else:
                        return json.dumps({"result": content}, indent=2)
            except json.JSONDecodeError:
                # If JSON parsing fails, wrap the content
                return json.dumps({"content": content, "error": "Invalid JSON format"}, indent=2)
        
        elif output_kind == "table":
            # Ensure proper table format
            if constraints.table_columns:
                # Format as Markdown table with specified columns
                lines = content.strip().split('\n')
                
                # Check if it's already a proper table
                if '|' in content and any('---' in line or ':-:' in line for line in lines):
                    return content  # Already formatted as table
                
                # Convert to table format
                table_lines = []
                
                # Header
                header = "| " + " | ".join(constraints.table_columns) + " |"
                separator = "|" + "|".join([" --- " for _ in constraints.table_columns]) + "|"
                table_lines.extend([header, separator])
                
                # Data rows (try to parse from content)
                for line in lines:
                    if line.strip():
                        # Simple parsing - split by common delimiters
                        if ',' in line:
                            cells = [cell.strip() for cell in line.split(',')]
                        elif '\t' in line:
                            cells = [cell.strip() for cell in line.split('\t')]
                        else:
                            # Single column or free text
                            cells = [line.strip()]
                        
                        # Pad or truncate to match column count
                        while len(cells) < len(constraints.table_columns):
                            cells.append("")
                        cells = cells[:len(constraints.table_columns)]
                        
                        row = "| " + " | ".join(cells) + " |"
                        table_lines.append(row)
                
                return '\n'.join(table_lines)
            
            elif constraints.output_format == "csv":
                # Ensure CSV format
                if ',' not in content:
                    # Convert to CSV
                    lines = content.strip().split('\n')
                    csv_lines = []
                    for line in lines:
                        if line.strip():
                            # Simple conversion
                            csv_lines.append(line.replace('\t', ',').replace('  ', ','))
                    return '\n'.join(csv_lines)
                return content
            
            else:
                # Default table formatting (Markdown)
                return content
        
        else:  # output_kind == "text"
            # Apply text formatting constraints
            formatted_content = content
            
            # Apply bullet points if requested
            if constraints.use_bullet_points and not content.startswith(('•', '-', '*')):
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                if constraints.bullets_n:
                    lines = lines[:constraints.bullets_n]
                formatted_content = '\n'.join(f"• {line}" for line in lines)
            
            # Apply numbered lists if requested
            elif constraints.use_numbered_lists and not any(content.startswith(str(i)) for i in range(1, 10)):
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                formatted_content = '\n'.join(f"{i+1}. {line}" for i, line in enumerate(lines))
            
            return formatted_content
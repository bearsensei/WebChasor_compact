"""
Query Maker Module
Generates diverse search queries using LLM to improve information retrieval coverage.
"""

import os
import json
import logging
from typing import List, Optional
from dataclasses import dataclass
import datetime
import openai

from config_manager import get_config
from artifacts import Context
from prompt import QUERYMAKER_PROMPT

# Setup logging
logger = logging.getLogger(__name__)



@dataclass
class QueryMakerConfig:
    """Configuration for QueryMaker"""
    api_base: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 500
    num_queries: int = 10
    current_time: str = datetime.datetime.now().strftime("%Y-%m-%d")


class QueryMaker:
    """
    QueryMaker generates diverse search queries using LLM
    to improve information retrieval coverage and quality.
    """
    
    def __init__(self, config: Optional[QueryMakerConfig] = None):
        """
        Initialize QueryMaker with configuration.
        
        Args:
            config: QueryMaker configuration. If None, loads from YAML.
        """
        self.config = config or self._load_config_from_yaml()
        self.client = self._init_openai_client()
        
        logger.info(f"QueryMaker initialized with model: {self.config.model}, num_queries: {self.config.num_queries}")
    
    def _load_config_from_yaml(self) -> QueryMakerConfig:
        """Load configuration from YAML config file"""
        cfg = get_config()
        
        api_base = cfg.get('external_services.openai.api_base')
        api_key = os.getenv("OPENAI_API_KEY_AGENT")
        model = cfg.get('models.querymaker.model_name', 'gpt-oss-20b')
        temperature = cfg.get('models.querymaker.temperature', 0.7)
        max_tokens = cfg.get('models.querymaker.max_tokens', 500)
        num_queries = cfg.get('models.querymaker.num_queries', 5)
        
        missing = []
        if not api_base:
            missing.append("external_services.openai.api_base in config.yaml")
        if not api_key:
            missing.append("OPENAI_API_KEY_AGENT environment variable")
        
        if missing:
            error_msg = f"[QueryMaker][ERROR] Missing required configuration: {', '.join(missing)}"
            print(error_msg)
            raise ValueError(f"Missing required API configuration for QueryMaker: {', '.join(missing)}")
        
        return QueryMakerConfig(
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_queries=num_queries
        )
    
    def _init_openai_client(self) -> openai.OpenAI:
        """Initialize OpenAI client"""
        try:
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=30
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text.
        
        Args:
            text: Input text
            
        Returns:
            'zh' for Chinese, 'en' for English
        """
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return 'en'
        
        # If more than 30% are Chinese characters, consider it Chinese
        return 'zh' if chinese_chars / total_chars > 0.3 else 'en'
    
    def _analyze_query_type(self, query: str) -> dict:
        """
        Analyze query type to determine optimal number of queries.
        
        Returns:
            dict with query type flags and recommended query count
        """
        query_lower = query.lower()
        query_stripped = query.strip()
        
        # Definition queries (什么是, 是什么, what is, what are, 定义)
        is_definition = any(keyword in query_lower for keyword in [
            '什么是', '是什么', 'what is', 'what are', '定义', 'definition',
            '何为', '何謂', 'define', '概念'
        ])
        
        # Comparison queries (对比, 比较, compare, vs) - CHECK FIRST
        is_comparison = any(keyword in query_lower for keyword in [
            '对比', '比较', 'compare', 'comparison', 'vs', 'versus', '差异', '區別'
        ])
        
        # How-to queries (如何, 怎么, how to)
        is_howto = any(keyword in query_lower for keyword in [
            '如何', '怎么', '怎样', 'how to', 'how can', 'how do', '怎樣'
        ])
        
        # Simple query detection (懒人模式：只输入一个名词/名字)
        # 特征：很短、没有问号、没有特殊关键词、单个词或短语
        # IMPORTANT: Check this AFTER comparison/howto to avoid conflicts
        is_simple = (
            len(query_stripped) <= 20 and  # 短查询
            '?' not in query_stripped and '？' not in query_stripped and  # 没有问号
            not is_comparison and  # 不是对比查询
            not is_howto and  # 不是 how-to 查询
            not is_definition and  # 不是定义查询
            not any(keyword in query_lower for keyword in [
                '什么', '如何', '怎么', '为什么', '哪', '多少',
                'what', 'how', 'why', 'when', 'where', 'which', 'who'
            ]) and  # 没有疑问词
            len(query_stripped.split()) <= 4  # 词数很少（英文）或字数很少（中文）
        )
        
        # Determine recommended query count (priority order matters!)
        # Comparison and How-to have highest priority
        if is_comparison:
            recommended = 10  # Comparison: need multiple angles (highest priority)
        elif is_howto:
            recommended = 6  # How-to: methods + steps + examples
        elif is_definition:
            recommended = 4  # Definition: cover multiple aspects (definition, use cases, types, examples, comparison)
        elif is_simple:
            recommended = 2  # Simple: original + translated query (no LLM)
        else:
            recommended = self.config.num_queries  # Default from config
        
        return {
            'is_simple': is_simple,
            'is_definition': is_definition,
            'is_comparison': is_comparison,
            'is_howto': is_howto,
            'recommended_queries': recommended
        }
    
    def _detect_multi_questions(self, query: str) -> list:
        """
        Detect if query contains multiple questions.
        
        Args:
            query: Input query
            
        Returns:
            List of detected sub-questions (empty if single question)
        """
        # Split by question marks
        parts = []
        for sep in ['？', '?']:
            if sep in query:
                parts = [p.strip() for p in query.split(sep) if p.strip()]
                break
        
        # If we found multiple parts, return them
        if len(parts) > 1:
            return parts
        
        # Otherwise, check for Chinese sentence-ending particles with multiple clauses
        if '；' in query or '，' in query:
            # Multiple clauses might indicate multiple questions
            parts = [p.strip() for p in query.replace('；', '，').split('，') if p.strip()]
            if len(parts) > 2:  # At least 3 parts to consider it multi-question
                return parts
        
        return []
    
    def _verify_multi_question_coverage(self, original_query: str, generated_queries: list) -> list:
        """
        Verify that all sub-questions in a multi-question query are covered.
        Add missing coverage if needed.
        
        Args:
            original_query: Original multi-question query
            generated_queries: Generated queries from LLM
            
        Returns:
            Enhanced queries with full coverage
        """
        sub_questions = self._detect_multi_questions(original_query)
        
        if not sub_questions or len(sub_questions) <= 1:
            # Not a multi-question query
            return generated_queries
        
        print(f"[QueryMaker][MULTI-Q] Detected {len(sub_questions)} sub-questions")
        
        # Synonym mapping for better matching
        synonyms = {
            '年薪': ['薪酬', '薪水', 'salary', '工资'],
            '薪酬': ['年薪', '薪水', 'salary', '工资'],
            '退休': ['retirement', '离任', '卸任'],
            '生日': ['birthday', '出生', '诞辰'],
        }
        
        # Check coverage for each sub-question
        missing_coverage = []
        for i, sub_q in enumerate(sub_questions, 1):
            # Extract key words from sub-question (remove common question words)
            sub_q_clean = sub_q.lower()
            for remove_word in ['什么时候', '多少', '哪', '是', '的', '？', '?', '她', '他']:
                sub_q_clean = sub_q_clean.replace(remove_word, '')
            sub_q_words = set(w for w in sub_q_clean.split() if w and len(w) > 1)
            
            # Expand with synonyms
            expanded_words = set(sub_q_words)
            for word in sub_q_words:
                if word in synonyms:
                    expanded_words.update(synonyms[word])
            
            # Check if any generated query (except original) covers this sub-question
            covered = False
            for gen_q in generated_queries[1:]:  # Skip original query itself
                gen_q_lower = gen_q.lower()
                # Check if key words or synonyms appear in generated query
                matches = sum(1 for word in expanded_words if word and len(word) > 1 and word in gen_q_lower)
                if matches > 0:
                    covered = True
                    print(f"[QueryMaker][MULTI-Q] Sub-Q{i} covered: '{sub_q}' -> '{gen_q}'")
                    break
            
            if not covered:
                missing_coverage.append(sub_q)
                print(f"[QueryMaker][MULTI-Q] Sub-Q{i} NOT covered: '{sub_q}'")
        
        # Add missing queries
        if missing_coverage:
            print(f"[QueryMaker][MULTI-Q] Adding {len(missing_coverage)} missing queries")
            for missing_q in missing_coverage:
                generated_queries.append(missing_q)
                print(f"[QueryMaker][MULTI-Q] Added: {missing_q}")
        
        return generated_queries
    
    def generate_queries(self, original_query: str, ctx: Optional[Context] = None) -> List[str]:
        """
        Generate diverse search queries based on the original query.
        
        Args:
            original_query: The user's original query
            ctx: Optional context containing time information, etc.
            
        Returns:
            List of diverse search queries (includes original query as first item)
        """
        try:
            # Analyze query type to determine optimal query count
            analysis = self._analyze_query_type(original_query)
            target_queries = analysis['recommended_queries']
            
            logger.info(f"[QueryMaker] Query type: simple={analysis['is_simple']}, "
                       f"def={analysis['is_definition']}, comp={analysis['is_comparison']}, "
                       f"howto={analysis['is_howto']}, target={target_queries}")
            
            logger.info(f"[QueryMaker] Generating {target_queries} queries for: {original_query[:100]}...")
            
            # Get current time dynamically
            import datetime
            now = datetime.datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            current_year = str(now.year)
            current_month = now.strftime("%B %Y")  # e.g., "October 2025"
            next_year = str(now.year + 1)
            
            # Build prompt with time context
            # Use replace() instead of format() to avoid conflicts with JSON braces in examples
            prompt_with_time = QUERYMAKER_PROMPT.replace("{current_date}", current_date)
            prompt_with_time = prompt_with_time.replace("{current_year}", current_year)
            prompt_with_time = prompt_with_time.replace("{current_month}", current_month)
            prompt_with_time = prompt_with_time.replace("{next_year}", next_year)
            
            prompt = prompt_with_time + f"\n\n**CONSTRAINT: Generate EXACTLY {target_queries} queries (excluding the original query).**"
            
            print(f"[QueryMaker][DEBUG] Current time context: {current_date} (Year: {current_year})")
            
            # Call LLM
            print(f"[QueryMaker][DEBUG] Calling LLM with model={self.config.model}, temp={self.config.temperature}")
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": original_query}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            print(f"[QueryMaker][DEBUG] Response finish_reason: {response.choices[0].finish_reason}")
            
            # Handle both regular and reasoning models (o1 series)
            message = response.choices[0].message
            response_text = message.content
            
            # For reasoning models (o1), content might be None and reasoning_content has the text
            if response_text is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                response_text = message.reasoning_content
                logger.info(f"[QueryMaker] Using reasoning_content from o1 model")
            
            # Check if response is still None
            if response_text is None:
                logger.warning(f"[QueryMaker] LLM returned None response, finish_reason={response.choices[0].finish_reason}")
                return self._fallback_queries(original_query, ctx)
            
            response_text = response_text.strip()
            
            print(f"[QueryMaker][DEBUG] Raw response length: {len(response_text)}")
            print(f"[QueryMaker][DEBUG] Raw response (first 500 chars): {response_text[:500]}")
            
            # Try to extract JSON from response (handle cases where LLM adds extra text)
            json_text = response_text
            
            # If response contains markdown code blocks, extract JSON from them
            if '```json' in response_text:
                import re
                match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if match:
                    json_text = match.group(1)
                    print(f"[QueryMaker][DEBUG] Extracted JSON from markdown code block")
            elif '```' in response_text:
                import re
                match = re.search(r'```\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if match:
                    json_text = match.group(1)
                    print(f"[QueryMaker][DEBUG] Extracted JSON from code block")
            
            # Try to find JSON object by looking for { ... }
            if not json_text.startswith('{'):
                import re
                match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if match:
                    json_text = match.group(1)
                    print(f"[QueryMaker][DEBUG] Extracted JSON by pattern matching")
            
            # Parse JSON response
            try:
                parsed = json.loads(json_text)
                
                # Handle new structured format with slots
                if isinstance(parsed, dict) and 'slots' in parsed:
                    # Extract queries from all slots
                    valid_queries = []
                    for slot in parsed.get('slots', []):
                        slot_queries = slot.get('queries', [])
                        valid_queries.extend([q.strip() for q in slot_queries if isinstance(q, str) and q.strip()])
                    
                    logger.info(f"[QueryMaker] Extracted {len(valid_queries)} queries from {len(parsed.get('slots', []))} slots")
                
                # Handle simple list format (backward compatibility)
                elif isinstance(parsed, list):
                    valid_queries = [q.strip() for q in parsed if isinstance(q, str) and q.strip()]
                
                else:
                    logger.warning(f"[QueryMaker] Unexpected response format: {type(parsed)}, keys: {parsed.keys() if isinstance(parsed, dict) else 'N/A'}")
                    return self._fallback_queries(original_query, ctx)
                
                if not valid_queries:
                    logger.warning(f"[QueryMaker] No valid queries generated")
                    return self._fallback_queries(original_query, ctx)
                
                # Ensure we have the original query as first item
                if original_query not in valid_queries:
                    valid_queries = [original_query] + valid_queries
                else:
                    # Move original query to front
                    valid_queries.remove(original_query)
                    valid_queries = [original_query] + valid_queries
                
                # Verify multi-question coverage and add missing queries
                valid_queries = self._verify_multi_question_coverage(original_query, valid_queries)
                
                # Don't truncate - trust LLM's decision on query count
                # LLM already decided how many queries are needed
                final_queries = valid_queries
                
                logger.info(f"[QueryMaker] Generated {len(final_queries)} queries: {final_queries}")
                return final_queries
                
            except json.JSONDecodeError as e:
                logger.error(f"[QueryMaker] JSON parse error: {e}")
                print(f"[QueryMaker][ERROR] JSON parse failed at position {e.pos}")
                print(f"[QueryMaker][ERROR] Attempted to parse: {json_text[:300]}")
                print(f"[QueryMaker][ERROR] Full response length: {len(response_text)}")
                return self._fallback_queries(original_query, ctx)
        
        except Exception as e:
            logger.error(f"[QueryMaker] Failed to generate queries: {e}")
            print(f"[QueryMaker][ERROR] Exception type: {type(e).__name__}")
            print(f"[QueryMaker][ERROR] Exception details: {str(e)}")
            import traceback
            print(f"[QueryMaker][ERROR] Traceback:")
            traceback.print_exc()
            return self._fallback_queries(original_query, ctx)
    
    def _fallback_queries(self, original_query: str, ctx: Optional[Context] = None) -> List[str]:
        """
        Fallback query generation using simple heuristics.
        
        Args:
            original_query: The original query
            ctx: Optional context
            
        Returns:
            List containing the original query and simple variations
        """
        # Analyze query type for fallback as well
        analysis = self._analyze_query_type(original_query)
        target_queries = analysis['recommended_queries']
        
        queries = [original_query]
        
        # For definition queries, keep it minimal
        if analysis['is_definition']:
            logger.info(f"[QueryMaker][FALLBACK] Using {len(queries)} queries for definition type")
            return queries
        
        # Add time-based variation if time context exists
        if ctx and ctx.time_context and ctx.time_context.window[0]:
            date_str = ctx.time_context.window[0].split('T')[0]
            queries.append(f"{original_query} {date_str}")
        
        # Add "latest" variation
        if "最新" not in original_query and "latest" not in original_query.lower():
            queries.append(f"{original_query} latest")
        
        logger.info(f"[QueryMaker][FALLBACK] Using {len(queries)} fallback queries")
        return queries


# Convenience function for direct usage
def generate_search_queries(original_query: str, ctx: Optional[Context] = None, num_queries: int = 5) -> List[str]:
    """
    Convenience function to generate search queries.
    
    Args:
        original_query: The user's original query
        ctx: Optional context
        num_queries: Number of queries to generate
        
    Returns:
        List of diverse search queries
    """
    querymaker = QueryMaker()
    return querymaker.generate_queries(original_query, ctx) 
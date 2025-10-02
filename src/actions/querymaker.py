"""
Query Maker Module
Generates diverse search queries using LLM to improve information retrieval coverage.
"""

import os
import json
import logging
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import openai

from config_manager import get_config
from artifacts import Context

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class QueryMakerConfig:
    """Configuration for QueryMaker"""
    api_base: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 500
    num_queries: int = 5


QUERYMAKER_PROMPT = """You are a search query generator. Given a user's original query, generate {num_queries} diverse search queries that will help retrieve comprehensive information to answer the user's question in the SAME LANGUAGE as the original query.

**CRITICAL: Output ONLY the JSON array. Do NOT output any thinking process, explanations, or commentary. Start your response directly with [ and end with ].**

## Requirements:
1. Do not just paraphrase the original query. Go beyond it.  
2. Include queries that are not easy to think of at first glance.  
3. Expand beyond the mentioned entity:  
   - Related entities (allies, competitors, collaborators, alternative systems)  
   - Opposite entities (contrastive cases, counterexamples, rivals)  
   - Entities not mentioned in the query but logically connected (historical figures, organizations, datasets, benchmarks, regions, etc.)  
4. Mix different perspectives: history, future, comparisons, failures, user experience, numbers/statistics, analogies.  
5. At least 2 queries should introduce new entities not present in the original query.  
6. Keep queries short (≤12 words) and suitable for search engines.  

## Examples:

[USER QUERY] 如何减少大模型的幻觉?
[OUTPUT]
["大模型幻觉的历史案例", "减少幻觉的未来研究方向", "OpenAI 与 Anthropic 对比 幻觉处理", "用户角度如何感知模型幻觉", "大模型幻觉率统计 2024"]

[USER QUERY] 如何减少大模型的幻觉?
[OUTPUT]
["为什么幻觉难以避免？哪些情况下幻觉有用？", "医学中减少幻觉的方法 与 AI 对比", "人类记忆错误 vs AI 幻觉", "业界实践 幻觉控制 案例 study"]

[USER QUERY] 介绍一下叶玉如校长
[OUTPUT]
["叶玉如 香港科技大学", "Nancy Ip Yuk-yu profile", "叶玉如 神经科学研究", "叶玉如 学术成就", "Professor Nancy Ip biography"]

## ACTUAL TASK
[USER QUERY] {user_query}
[OUTPUT]
"""


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
        
        if not all([api_base, api_key]):
            raise ValueError("Missing required API configuration for QueryMaker")
        
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
            logger.info(f"[QueryMaker] Generating {self.config.num_queries} queries for: {original_query[:100]}...")
            
            # Build prompt
            prompt = QUERYMAKER_PROMPT.format(
                num_queries=self.config.num_queries,
                user_query=original_query
            )
            
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
            
            # Parse JSON response
            try:
                queries = json.loads(response_text)
                
                if not isinstance(queries, list):
                    logger.warning(f"[QueryMaker] Response is not a list: {response_text[:100]}")
                    return self._fallback_queries(original_query, ctx)
                
                # Filter and validate queries
                valid_queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
                
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
                
                # Limit to configured number
                final_queries = valid_queries[:self.config.num_queries]
                
                logger.info(f"[QueryMaker] Generated {len(final_queries)} queries: {final_queries}")
                return final_queries
                
            except json.JSONDecodeError as e:
                logger.error(f"[QueryMaker] JSON parse error: {e}, response: {response_text[:200]}")
                return self._fallback_queries(original_query, ctx)
        
        except Exception as e:
            logger.error(f"[QueryMaker] Failed to generate queries: {e}")
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
        queries = [original_query]
        
        # Add time-based variation if time context exists
        if ctx and ctx.time_context and ctx.time_context.window[0]:
            date_str = ctx.time_context.window[0].split('T')[0]
            queries.append(f"{original_query} {date_str}")
        
        # Add "latest" variation
        if "最新" not in original_query and "latest" not in original_query.lower():
            queries.append(f"{original_query} latest")
        
        logger.info(f"[QueryMaker][FALLBACK] Using {len(queries)} fallback queries")
        return queries[:self.config.num_queries]


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
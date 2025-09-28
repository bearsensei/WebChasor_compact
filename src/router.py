"""
Router module for Chasor system.
Classifies user queries into appropriate categories using hybrid approach:
1. Fast heuristics for high-precision early exits
2. Lightweight classifier for common cases  
3. LLM fallback for low-confidence cases
"""

import os
import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import openai

from prompt import ROUTER_PROMPT, USER_PROMPT, SYSTEM_PROMPT_MULTI

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class QueryCategory(Enum):
    """Enumeration of supported query categories"""
    INFORMATION_RETRIEVAL = "INFORMATION_RETRIEVAL"
    MATH_QUERY = "MATH_QUERY"
    TASK_PRODUCTIVITY = "TASK_PRODUCTIVITY"
    KNOWLEDGE_REASONING = "KNOWLEDGE_REASONING"
    CONVERSATIONAL_FOLLOWUP = "CONVERSATIONAL_FOLLOWUP"
    CREATIVE_GENERATION = "CREATIVE_GENERATION"
    MULTIMODAL_QUERY = "MULTIMODAL_QUERY"

@dataclass
class RouterSignals:
    """Signals extracted from query analysis"""
    has_media: bool
    needs_external_facts: bool
    numeric_only: bool

@dataclass
class RouterResult:
    """Structured router output"""
    label: str
    confidence: float
    signals: RouterSignals
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "signals": asdict(self.signals)
        }

@dataclass
class RouterConfig:
    """Configuration for the Router"""
    api_base: str
    api_key: str
    model: str
    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.1
    llm_confidence_threshold: float = 0.55

class Router:
    """
    Hybrid query router that uses fast heuristics, lightweight classification,
    and LLM fallback to classify user queries.
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        """
        Initialize Router with configuration.
        
        Args:
            config: Router configuration. If None, loads from environment.
        """
        self.config = config or self._load_config_from_env()
        self.client = self._init_openai_client()
        
        # Initialize lightweight classifier (placeholder for now)
        self._init_lightweight_classifier()
        
        logger.info(f"Router initialized with model: {self.config.model}")
    
    def _load_config_from_env(self) -> RouterConfig:
        """Load configuration from environment variables"""
        api_base = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY_AGENT")
        model = os.getenv("OPENAI_API_MODEL_AGENT_ROUTER")
        
        if not all([api_base, api_key, model]):
            missing = [k for k, v in {
                "OPENAI_API_BASE": api_base,
                "OPENAI_API_KEY_AGENT": api_key,
                "OPENAI_API_MODEL_AGENT_ROUTER": model
            }.items() if not v]
            raise ValueError(f"Missing required environment variables: {missing}")
        
        return RouterConfig(
            api_base=api_base,
            api_key=api_key,
            model=model
        )
    
    def _init_openai_client(self) -> openai.OpenAI:
        """Initialize OpenAI client with configuration"""
        try:
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _init_lightweight_classifier(self):
        """Initialize lightweight classifier (placeholder)"""
        # TODO: Initialize actual lightweight classifier (sklearn, etc.)
        # For now, use rule-based classification
        self.labels = [cat.value for cat in QueryCategory]
        logger.info("Lightweight classifier initialized (rule-based)")
    
    async def classify(self, history: str, user_query: str) -> QueryCategory:
        """
        Classify a user query (backward compatibility).
        
        Args:
            history: Previous conversation history
            user_query: Current user query to classify
            
        Returns:
            QueryCategory: The classified category
        """
        result = await self.route(user_query, has_media=False, last_assistant_turn=history)
        return QueryCategory(result.label)
    
    async def route(self, query: str, has_media: bool = False, last_assistant_turn: str = "") -> RouterResult:
        """
        Route query using hybrid approach: heuristics -> classifier -> LLM fallback.
        
        Args:
            query: User query to classify
            has_media: Whether query contains media (images, audio, etc.)
            last_assistant_turn: Previous assistant response for context
            
        Returns:
            RouterResult: Structured routing result
        """
        try:
            logger.info(f"Routing query: {query[:100]}...")
            
            # Extract signals first
            signals = self._extract_signals(query, has_media)
            
            # 1) Fast heuristics for high-precision early exits
            heuristic_result = self._apply_heuristics(query, has_media)
            if heuristic_result:
                label, confidence = heuristic_result
                logger.info(f"Heuristic match: {label} (confidence: {confidence})")
                return RouterResult(label=label, confidence=confidence, signals=signals)
            
            # 2) Lightweight classifier
            classifier_result = self._lightweight_classify(query, last_assistant_turn)
            label, confidence = classifier_result
            
            # 3) Low-confidence fallback to LLM
            if confidence < self.config.llm_confidence_threshold:
                logger.info(f"Low confidence ({confidence}), using LLM fallback")
                llm_label = await self._llm_router(query, last_assistant_turn)
                label = llm_label
                confidence = 0.65  # Conservative default for LLM
            
            logger.info(f"Final classification: {label} (confidence: {confidence})")
            return RouterResult(label=label, confidence=confidence, signals=signals)
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            # Return safe fallback
            return RouterResult(
                label=QueryCategory.INFORMATION_RETRIEVAL.value,
                confidence=0.3,
                signals=RouterSignals(has_media=has_media, needs_external_facts=True, numeric_only=False)
            )
    
    def _extract_signals(self, query: str, has_media: bool) -> RouterSignals:
        """Extract signals from query analysis"""
        # Analyze query for various signals
        needs_external_facts = self._needs_external_facts(query)
        numeric_only = self._is_numeric_only(query)
        
        return RouterSignals(
            has_media=has_media,
            needs_external_facts=needs_external_facts,
            numeric_only=numeric_only
        )
    
    def _needs_external_facts(self, query: str) -> bool:
        """Determine if query needs external facts/information"""
        external_indicators = [
            r'\b(what|when|where|who|how|why)\b',
            r'\b(current|latest|recent|today|now)\b',
            r'\b(weather|news|price|stock|rate)\b',
            r'\b(search|find|lookup|check)\b'
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in external_indicators)
    
    def _is_numeric_only(self, query: str) -> bool:
        """Determine if query is purely numeric/mathematical"""
        # Remove spaces and check if mostly numbers and math operators
        clean_query = re.sub(r'\s+', '', query)
        math_chars = set('0123456789+-*/()=.^%√∫∑')
        query_chars = set(clean_query)
        
        if len(clean_query) == 0:
            return False
        
        # If >80% of characters are math-related, consider it numeric only
        math_ratio = len(query_chars & math_chars) / len(query_chars)
        return math_ratio > 0.8
    
    def _apply_heuristics(self, query: str, has_media: bool) -> Optional[Tuple[str, float]]:
        """Apply fast heuristics for high-precision early exits"""
        
        # 1) Media detection
        if has_media:
            return QueryCategory.MULTIMODAL_QUERY.value, 0.99
        
        # 2) Math detection
        if self._looks_like_math(query):
            return QueryCategory.MATH_QUERY.value, 0.98
        
        # 3) Task/productivity detection
        if self._asks_for_rewrite_or_format(query):
            return QueryCategory.TASK_PRODUCTIVITY.value, 0.95
        
        # 4) Creative generation detection
        if self._asks_for_poem_story_roleplay(query):
            return QueryCategory.CREATIVE_GENERATION.value, 0.95
        
        return None
    
    def _looks_like_math(self, query: str) -> bool:
        """Detect mathematical queries"""
        math_patterns = [
            r'\b(calculate|compute|solve|find)\s+.*\d',
            r'\d+\s*[\+\-\*\/\^]\s*\d+',
            r'\b(area|volume|perimeter|circumference|derivative|integral)\b',
            r'\b(sin|cos|tan|log|ln|sqrt|factorial)\s*\(',
            r'\b\d+\s*(plus|minus|times|divided by)\s*\d+\b'
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in math_patterns)
    
    def _asks_for_rewrite_or_format(self, query: str) -> bool:
        """Detect task/productivity queries"""
        task_patterns = [
            r'\b(rewrite|rephrase|format|organize|structure)\b',
            r'\b(summarize|outline|bullet points|list)\b',
            r'\b(translate|convert|transform)\b',
            r'\b(edit|revise|improve|optimize)\b'
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in task_patterns)
    
    def _asks_for_poem_story_roleplay(self, query: str) -> bool:
        """Detect creative generation queries"""
        creative_patterns = [
            r'\b(write|create|generate).*\b(poem|story|song|lyrics)\b',
            r'\b(roleplay|role.play|pretend|act as)\b',
            r'\b(creative|imaginative|fictional)\b',
            r'\b(character|dialogue|narrative|plot)\b'
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in creative_patterns)
    
    def _lightweight_classify(self, query: str, last_assistant_turn: str) -> Tuple[str, float]:
        """Lightweight classification using simple rules/features"""
        # TODO: Replace with actual ML classifier
        # For now, use rule-based classification with confidence scores
        
        features = self._extract_features(query, last_assistant_turn)
        
        # Simple rule-based classification
        if features['question_words'] > 0 and features['external_indicators'] > 0:
            return QueryCategory.INFORMATION_RETRIEVAL.value, 0.75
        elif features['math_indicators'] > 0:
            return QueryCategory.MATH_QUERY.value, 0.70
        elif features['task_indicators'] > 0:
            return QueryCategory.TASK_PRODUCTIVITY.value, 0.65
        elif features['creative_indicators'] > 0:
            return QueryCategory.CREATIVE_GENERATION.value, 0.60
        elif features['followup_indicators'] > 0:
            return QueryCategory.CONVERSATIONAL_FOLLOWUP.value, 0.55
        else:
            return QueryCategory.KNOWLEDGE_REASONING.value, 0.45
    
    def _extract_features(self, query: str, last_assistant_turn: str) -> Dict[str, int]:
        """Extract features for lightweight classification"""
        query_lower = query.lower()
        
        features = {
            'question_words': len(re.findall(r'\b(what|when|where|who|how|why)\b', query_lower)),
            'external_indicators': len(re.findall(r'\b(current|latest|search|find|weather|news)\b', query_lower)),
            'math_indicators': len(re.findall(r'\b(calculate|solve|math|number|\d+)\b', query_lower)),
            'task_indicators': len(re.findall(r'\b(write|create|format|organize|list)\b', query_lower)),
            'creative_indicators': len(re.findall(r'\b(story|poem|creative|imagine|roleplay)\b', query_lower)),
            'followup_indicators': len(re.findall(r'\b(also|additionally|furthermore|what about)\b', query_lower)),
        }
        
        return features
    
    async def _llm_router(self, query: str, last_assistant_turn: str) -> str:
        """LLM fallback for low-confidence cases"""
        try:
            # Prepare context
            context = ""
            if last_assistant_turn and last_assistant_turn.strip():
                context = f"Previous assistant response: {last_assistant_turn}\n\n"
            
            user_content = f"{context}User query: {query}"
            
            messages = [
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=100
            )
            
            raw_response = response.choices[0].message.content
            
            # Extract category from response
            response_upper = raw_response.upper()
            for category in QueryCategory:
                if category.value in response_upper:
                    return category.value
            
            # Fallback to information retrieval
            logger.warning(f"LLM router couldn't parse response: {raw_response}")
            return QueryCategory.INFORMATION_RETRIEVAL.value
            
        except Exception as e:
            logger.error(f"LLM router failed: {e}")
            return QueryCategory.INFORMATION_RETRIEVAL.value
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the router"""
        health = {
            "status": "unknown",
            "model": self.config.model,
            "api_base": self.config.api_base,
            "categories_supported": len(self.labels),
            "llm_threshold": self.config.llm_confidence_threshold
        }
        
        try:
            # Test with a simple query
            test_result = await self.route("Hello, how are you?")
            
            health["status"] = "healthy"
            health["test_result"] = test_result.to_dict()
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            logger.error(f"Router health check failed: {e}")
        
        return health

# Example usage and testing
async def test_router():
    """Test the router with sample queries"""
    router = Router()
    
    test_cases = [
        ("What are the strengths and weaknesses of renewable energy?", False, ""),
        ("Calculate the area of a circle with radius 5", False, ""),
        ("Write a creative story about space exploration", False, ""),
        ("Can you also tell me about Mars?", False, "I told you about Earth's atmosphere."),
        ("2 + 2 = ?", False, ""),
    ]
    
    print("=== Router Test Results ===")
    for query, has_media, last_turn in test_cases:
        try:
            result = await router.route(query, has_media, last_turn)
            print(f"Query: {query}")
            print(f"Result: {result.to_dict()}")
            print("-" * 50)
        except Exception as e:
            print(f"Error routing '{query}': {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_router())
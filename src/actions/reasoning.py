"""
Reasoning Action for WebChasor System

Handles analytical queries that don't require external information retrieval.
Provides structured, thoughtful responses using reasoning scaffolds to ensure
clarity and prevent hallucination.

Examples:
- "Why do companies merge?"
- "What are the pros and cons of electric vehicles?"
- "How does inflation affect different economic sectors?"
"""

import os
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from dotenv import load_dotenv
import openai

from artifacts import Action, Artifact, Context
from config_manager import get_config

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    """Types of reasoning approaches"""
    ANALYTICAL = "analytical"          # Pros/cons, cause/effect analysis
    COMPARATIVE = "comparative"        # Compare multiple options/concepts
    EXPLANATORY = "explanatory"       # Explain how/why something works
    EVALUATIVE = "evaluative"         # Assess quality, effectiveness, value
    PREDICTIVE = "predictive"         # Analyze trends, future implications

@dataclass
class ReasoningScaffold:
    """Structure for reasoning scaffolds"""
    name: str
    steps: List[str]
    description: str
    suitable_for: List[str]

@dataclass
class ReasoningConfig:
    """Configuration for reasoning classification"""
    api_base: str
    api_key: str
    model: str
    timeout: int = 30
    temperature: float = 0.1
    use_model_classification: bool = True
    classification_confidence_threshold: float = 0.7

# Reasoning type classification prompt
REASONING_TYPE_PROMPT = """You are a reasoning type classifier. Analyze the user query and determine which type of reasoning approach would be most appropriate.

Available reasoning types:
- ANALYTICAL: For analyzing complex topics with multiple dimensions, cause-effect relationships, pros/cons analysis
- COMPARATIVE: For comparing multiple options, concepts, or approaches (vs, better/worse, advantages/disadvantages)
- EXPLANATORY: For explaining how things work, mechanisms, processes, or concepts
- EVALUATIVE: For assessing quality, effectiveness, value, or making judgments
- PREDICTIVE: For analyzing trends, future implications, impacts, or consequences

Respond with ONLY the reasoning type (e.g., "ANALYTICAL", "COMPARATIVE", etc.) followed by a confidence score (0.0-1.0).
Format: REASONING_TYPE confidence_score

Examples:
- "Why do companies merge?" → ANALYTICAL 0.9
- "Compare electric vs gas cars" → COMPARATIVE 0.95
- "How does photosynthesis work?" → EXPLANATORY 0.9
- "Is this marketing strategy effective?" → EVALUATIVE 0.85
- "What will be the impact of AI on jobs?" → PREDICTIVE 0.9"""

# Define different reasoning scaffolds
REASONING_SCAFFOLDS = {
    ReasoningType.ANALYTICAL: ReasoningScaffold(
        name="Analytical Reasoning",
        steps=[
            "Step 1: Restate the core question clearly",
            "Step 2: Identify 2-3 key dimensions or factors",
            "Step 3: Analyze each dimension (pros/cons/causal relationships)",
            "Step 4: Consider interconnections between factors",
            "Step 5: Synthesize the main insights",
            "Step 6: Discuss implications and limitations"
        ],
        description="For analyzing complex topics with multiple dimensions",
        suitable_for=["why questions", "cause-effect analysis", "multi-factor problems"]
    ),
    
    ReasoningType.COMPARATIVE: ReasoningScaffold(
        name="Comparative Analysis",
        steps=[
            "Step 1: Clearly define what is being compared",
            "Step 2: Establish comparison criteria or dimensions",
            "Step 3: Analyze similarities between options",
            "Step 4: Examine key differences",
            "Step 5: Evaluate trade-offs and contexts",
            "Step 6: Provide balanced conclusions"
        ],
        description="For comparing multiple options, concepts, or approaches",
        suitable_for=["vs comparisons", "choice decisions", "alternative analysis"]
    ),
    
    ReasoningType.EXPLANATORY: ReasoningScaffold(
        name="Explanatory Reasoning",
        steps=[
            "Step 1: Define the concept or phenomenon",
            "Step 2: Break down into component parts",
            "Step 3: Explain how components interact",
            "Step 4: Provide concrete examples",
            "Step 5: Address common misconceptions",
            "Step 6: Summarize key takeaways"
        ],
        description="For explaining how things work or why they happen",
        suitable_for=["how questions", "mechanism explanation", "concept clarification"]
    ),
    
    ReasoningType.EVALUATIVE: ReasoningScaffold(
        name="Evaluative Analysis",
        steps=[
            "Step 1: Define evaluation criteria",
            "Step 2: Assess strengths and advantages",
            "Step 3: Identify weaknesses and limitations",
            "Step 4: Consider context and constraints",
            "Step 5: Weigh evidence and trade-offs",
            "Step 6: Provide reasoned judgment"
        ],
        description="For assessing quality, effectiveness, or value",
        suitable_for=["evaluation questions", "quality assessment", "effectiveness analysis"]
    ),
    
    ReasoningType.PREDICTIVE: ReasoningScaffold(
        name="Predictive Analysis",
        steps=[
            "Step 1: Identify current state and trends",
            "Step 2: Analyze driving forces and factors",
            "Step 3: Consider potential scenarios",
            "Step 4: Assess probabilities and uncertainties",
            "Step 5: Discuss implications and consequences",
            "Step 6: Highlight key assumptions and limitations"
        ],
        description="For analyzing trends and future implications",
        suitable_for=["future trends", "impact analysis", "scenario planning"]
    )
}

class REASONING(Action):
    """
    Reasoning action that provides structured analytical responses
    for queries that don't require external information retrieval.
    """
    
    name = "REASONING"
    requires_tools = ["synthesizer"]
    
    def __init__(self, config: Optional[ReasoningConfig] = None):
        """Initialize the reasoning action"""
        super().__init__()
        self.config = config or self._load_config_from_env()
        self.client = self._init_openai_client() if self.config.use_model_classification else None
        logger.info(f"REASONING action initialized (model classification: {self.config.use_model_classification})")
    
    def _load_config_from_env(self) -> ReasoningConfig:
        """Load configuration from environment variables"""
        api_base = get_config().get('external_services.openai.api_base', 'https://api.openai.com/v1')
        api_key = os.getenv("OPENAI_API_KEY_AGENT")
        model = get_config().get('models.reasoning.model_name', 'gpt-4')
        
        # Model classification is optional - fallback to rule-based if not configured
        # Disable model classification for now to avoid API issues
        use_model = False  # Temporarily disable model classification
        
        return ReasoningConfig(
            api_base=api_base or "",
            api_key=api_key or "",
            model=model,
            use_model_classification=use_model
        )
    
    def _init_openai_client(self) -> Optional[openai.OpenAI]:
        """Initialize OpenAI client for model-based classification"""
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
        Execute reasoning analysis on the given query.
        
        Args:
            ctx: Context containing the query and metadata
            toolset: Available tools (requires synthesizer)
            
        Returns:
            Artifact: Structured reasoning response
        """
        try:
            logger.info(f"Starting reasoning analysis for: {ctx.query[:100]}...")
            
            # Validate inputs
            if not ctx.query or not ctx.query.strip():
                raise ValueError("Query cannot be empty")
            
            if not hasattr(toolset, 'synthesizer'):
                raise ValueError("Synthesizer tool is required but not available")
            
            # Determine appropriate reasoning type
            reasoning_type, confidence = await self._determine_reasoning_type(ctx.query)
            logger.info(f"Selected reasoning type: {reasoning_type.value} (confidence: {confidence})")
            
            # Get appropriate scaffold
            scaffold = REASONING_SCAFFOLDS[reasoning_type]
            
            # Determine which category config to use based on router_category
            # If it's CONVERSATIONAL_FOLLOWUP, use shorter response config
            cfg = get_config()
            
            if ctx.router_category == "CONVERSATIONAL_FOLLOWUP":
                # Use CONVERSATIONAL_FOLLOWUP config (300 tokens, brief)
                length_config = cfg.get_response_length_config("CONVERSATIONAL_FOLLOWUP")
                category_for_synthesis = "CONVERSATIONAL_FOLLOWUP"
                logger.info(f"Using CONVERSATIONAL_FOLLOWUP config for brief response")
            else:
                # Use KNOWLEDGE_REASONING config (6000 tokens, comprehensive)
                length_config = cfg.get_response_length_config("KNOWLEDGE_REASONING")
                category_for_synthesis = "KNOWLEDGE_REASONING"
            
            max_tokens = length_config.get('max_tokens', 6000)
            temperature = length_config.get('temperature', 0.7)
            
            logger.info(f"Response config: max_tokens={max_tokens}, temperature={temperature}")
            
            constraints = {
                "language": "auto",  # Let synthesizer detect language
                "tone": "friendly, conversational",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "instruction_hint": f"Focus on {reasoning_type.value} reasoning approach. Use diverse formatting: bullet points, numbered lists, comparison tables, and structured presentations where appropriate. Mix paragraphs with lists and tables for better readability."
            }
            
            # Execute reasoning through synthesizer using generate method
            text = await toolset.synthesizer.generate(
                category=category_for_synthesis,  # Use determined category
                style_key="auto",  # 使用 auto 启用自动语言检测和样式选择
                constraints=constraints,
                materials=ctx.query,  # Pass the original query as materials
                task_scaffold=None  # Let the hidden reasoning scaffold handle structure
            )
            
            # Validate output (but allow short responses for conversational followup)
            min_length = 10 if ctx.router_category == "CONVERSATIONAL_FOLLOWUP" else 50
            if not text or len(text.strip()) < min_length:
                logger.warning(f"Reasoning output seems too short ({len(text.strip()) if text else 0} chars, min: {min_length}), retrying...")
                # Retry with more explicit constraints
                constraints["instruction_hint"] = f"Provide a comprehensive {reasoning_type.value} explanation with concrete examples and practical insights."
                text = await toolset.synthesizer.generate(
                    category=category_for_synthesis,  # Use determined category
                    style_key="auto",  # 使用 auto 启用自动语言检测和样式选择
                    constraints=constraints,
                    materials=ctx.query,
                    task_scaffold=None
                )
            
            # Create artifact with metadata
            artifact = Artifact(
                kind="reasoning_analysis",
                content=text,
                meta={
                    "category": ctx.router_category,
                    "reasoning_type": reasoning_type.value,
                    "reasoning_confidence": confidence,
                    "scaffold_used": scaffold.name,
                    "query_length": len(ctx.query),
                    "response_length": len(text),
                    "steps_count": len(scaffold.steps),
                    "classification_method": "model" if self.client else "rule-based"
                }
            )
            
            logger.info(f"Reasoning analysis completed successfully ({len(text)} chars)")
            return artifact
            
        except Exception as e:
            logger.error(f"Reasoning analysis failed: {str(e)}")
            
            # Return fallback response
            fallback_text = self._generate_fallback_response(ctx.query, str(e))
            return Artifact(
                kind="reasoning_analysis",
                content=fallback_text,
                meta={
                    "category": ctx.router_category,
                    "error": str(e),
                    "fallback": True
                }
            )
    
    async def _determine_reasoning_type(self, query: str) -> tuple[ReasoningType, float]:
        """
        Determine the most appropriate reasoning type for the query.
        Uses model-based classification if available, falls back to rule-based.
        
        Args:
            query: User query to analyze
            
        Returns:
            tuple: (ReasoningType, confidence_score)
        """
        # Try model-based classification first
        if self.client and self.config.use_model_classification:
            try:
                model_result = await self._model_based_classification(query)
                if model_result:
                    reasoning_type, confidence = model_result
                    if confidence >= self.config.classification_confidence_threshold:
                        logger.info(f"Model classification: {reasoning_type.value} (confidence: {confidence})")
                        return reasoning_type, confidence
                    else:
                        logger.info(f"Model confidence too low ({confidence}), falling back to rules")
            except Exception as e:
                logger.warning(f"Model classification failed: {e}, falling back to rule-based")
        
        # Fallback to rule-based classification
        reasoning_type = self._rule_based_classification(query)
        confidence = 0.6  # Default confidence for rule-based
        logger.info(f"Rule-based classification: {reasoning_type.value} (confidence: {confidence})")
        
        # Simple decision print
        print(f"[REASONING][DECISION] {reasoning_type.value} conf={confidence} method=rule-based")
        
        return reasoning_type, confidence
    
    async def _model_based_classification(self, query: str) -> Optional[tuple[ReasoningType, float]]:
        """
        Use LLM to classify the reasoning type.
        
        Args:
            query: User query to classify
            
        Returns:
            Optional tuple of (ReasoningType, confidence) or None if failed
        """
        try:
            messages = [
                {"role": "system", "content": REASONING_TYPE_PROMPT},
                {"role": "user", "content": f"Query: {query}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=50
            )
            
            raw_response = response.choices[0].message.content
            
            # Check if response is valid
            if not raw_response or not raw_response.strip():
                logger.warning("Model returned empty response for reasoning classification")
                return None
            
            raw_response = raw_response.strip()
            logger.debug(f"Model classification response: {raw_response}")
            
            # Parse response (format: "REASONING_TYPE confidence_score")
            parts = raw_response.split()
            if len(parts) >= 2:
                type_str = parts[0].upper()
                confidence = float(parts[1])
                
                # Validate reasoning type
                for reasoning_type in ReasoningType:
                    if reasoning_type.value.upper() == type_str:
                        return reasoning_type, confidence
            
            # If parsing fails, try to extract just the type
            for reasoning_type in ReasoningType:
                if reasoning_type.value.upper() in raw_response.upper():
                    return reasoning_type, 0.7  # Default confidence
            
            logger.warning(f"Could not parse model response: {raw_response}")
            return None
            
        except Exception as e:
            logger.error(f"Model-based classification error: {e}")
            return None
    
    def _rule_based_classification(self, query: str) -> ReasoningType:
        """
        Rule-based fallback classification.
        
        Args:
            query: User query to analyze
            
        Returns:
            ReasoningType: Most suitable reasoning approach
        """
        query_lower = query.lower()
        
        # Comparative indicators
        comparative_keywords = [
            'vs', 'versus', 'compare', 'comparison', 'difference', 'better',
            'worse', 'advantages', 'disadvantages', 'pros and cons'
        ]
        if any(keyword in query_lower for keyword in comparative_keywords):
            return ReasoningType.COMPARATIVE
        
        # Explanatory indicators
        explanatory_keywords = [
            'how does', 'how do', 'explain', 'what is', 'what are',
            'mechanism', 'process', 'works', 'function'
        ]
        if any(keyword in query_lower for keyword in explanatory_keywords):
            return ReasoningType.EXPLANATORY
        
        # Evaluative indicators
        evaluative_keywords = [
            'evaluate', 'assess', 'judge', 'quality', 'effectiveness',
            'good', 'bad', 'successful', 'worth', 'value'
        ]
        if any(keyword in query_lower for keyword in evaluative_keywords):
            return ReasoningType.EVALUATIVE
        
        # Predictive indicators
        predictive_keywords = [
            'future', 'will', 'predict', 'trend', 'forecast',
            'impact', 'effect', 'consequence', 'implication'
        ]
        if any(keyword in query_lower for keyword in predictive_keywords):
            return ReasoningType.PREDICTIVE
        
        # Default to analytical
        return ReasoningType.ANALYTICAL
    
    def _build_reasoning_prompt(self, query: str, scaffold: ReasoningScaffold, ctx: Context) -> str:
        """
        Build a structured reasoning prompt using the appropriate scaffold.
        
        Args:
            query: User query
            scaffold: Reasoning scaffold to use
            ctx: Context for additional information
            
        Returns:
            str: Formatted reasoning prompt
        """
        # Include context hints if available
        context_info = ""
        if hasattr(ctx, 'hints') and ctx.hints:
            if 'remediation' in ctx.hints:
                context_info = f"\nPrevious attempt feedback: {ctx.hints['remediation']}\n"
        
        prompt = f"""Please provide a structured {scaffold.name.lower()} for the following question.

{context_info}
Question: {query}

Use this reasoning framework:

{chr(10).join(scaffold.steps)}

Guidelines:
- Be thorough but concise in each step
- Use clear, logical reasoning
- Avoid speculation or unsupported claims
- Provide specific examples where helpful
- Ensure each step builds on the previous ones
- Conclude with actionable insights where appropriate

Begin your analysis:"""

        return prompt
    
    def _build_enhanced_prompt(self, query: str, scaffold: ReasoningScaffold) -> str:
        """Build an enhanced prompt for retry attempts"""
        return f"""Please provide a comprehensive analysis of: {query}

This is a reasoning task that requires structured thinking. Please follow these steps carefully:

{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(scaffold.steps))}

Requirements:
- Each step should be at least 2-3 sentences
- Use concrete examples and evidence
- Show clear logical connections between ideas
- Avoid vague generalizations
- Provide a substantive conclusion

Please ensure your response is detailed and well-structured."""
    
    def _generate_fallback_response(self, query: str, error: str) -> str:
        """Generate a fallback response when reasoning fails"""
        return f"""I apologize, but I encountered an issue while analyzing your question: "{query}"

Error details: {error}

However, I can offer this basic analysis:

This appears to be a complex question that would benefit from structured reasoning. To properly address it, consider:

1. Breaking down the question into key components
2. Identifying the main factors or dimensions involved
3. Analyzing relationships and trade-offs
4. Considering different perspectives or stakeholders
5. Drawing evidence-based conclusions

For a more detailed analysis, please try rephrasing your question or providing additional context."""
    
    def get_available_scaffolds(self) -> Dict[str, ReasoningScaffold]:
        """Get all available reasoning scaffolds"""
        return {rt.value: scaffold for rt, scaffold in REASONING_SCAFFOLDS.items()}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the reasoning action"""
        health = {
            "status": "healthy",
            "scaffolds_available": len(REASONING_SCAFFOLDS),
            "reasoning_types": [rt.value for rt in ReasoningType],
            "requires_tools": self.requires_tools,
            "model_classification": self.config.use_model_classification,
            "model": self.config.model if self.config.use_model_classification else None
        }
        
        # Test model classification if available
        if self.client:
            try:
                test_result = await self._model_based_classification("How does photosynthesis work?")
                health["model_test"] = "passed" if test_result else "failed"
                if test_result:
                    health["model_test_result"] = {"type": test_result[0].value, "confidence": test_result[1]}
            except Exception as e:
                health["model_test"] = "failed"
                health["model_test_error"] = str(e)
        
        return health
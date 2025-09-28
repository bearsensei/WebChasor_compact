# Chasor is a system that can extract, calculate, synthesize, retrieve, and reorganize information. 
# Extractor is a class that can extract information from a text. Calculator is a class that can calculate information. 
# Synthesizer is a class that can synthesize information. InfoRetriever is a class that can retrieve information 
# from the search engine or a database. InfoReorganizer is a class that can reorganize the retrieved information 
# into a structured format.

# Architecture: router → registry → action.run → artifact

import os
import sys
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from artifacts import ActionRegistry, Context, Artifact, Report

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ExecutionState:
    """Track execution state across rounds"""
    rounds: int = 0
    plan: Optional[Any] = None
    extracted: Optional[Dict[str, Any]] = None
    artifacts: List[Artifact] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []

class ChasorCore:
    """
    Core Chasor system that orchestrates information extraction, calculation, 
    synthesis, retrieval, and reorganization through a multi-round execution loop.
    """
    
    def __init__(self, router, registry: ActionRegistry, toolset, evaluator, max_rounds: int = 2):
        """
        Initialize ChasorCore with required components.
        
        Args:
            router: Routes queries to appropriate actions
            registry: Registry of available actions
            toolset: Available tools for actions to use
            evaluator: Evaluates artifact quality
            max_rounds: Maximum execution rounds (default: 2)
        """
        self.router = router
        self.registry = registry
        self.toolset = toolset
        self.evaluator = evaluator
        self.max_rounds = max_rounds
        
        logger.info(f"ChasorCore initialized with max_rounds={max_rounds}")

    async def run(self, history: str, user_query: str) -> Artifact:
        """
        Main execution loop that processes user query through multiple rounds if needed.
        
        Args:
            history: Previous conversation history
            user_query: Current user query
            
        Returns:
            Artifact: Final processed result
            
        Raises:
            Exception: If execution fails after all rounds
        """
        try:
            logger.info(f"Starting execution for query: {user_query[:100]}...")
            
            # Route the query to appropriate category
            category = await self.router.classify(history, user_query)
            logger.info(f"Query classified as: {category}")
            
            # Initialize context and get action
            ctx = Context(
                history=history, 
                query=user_query, 
                router_category=category, 
                hints={}
            )
            action = self.registry.get(self.registry.route(category))
            
            if not action:
                raise ValueError(f"No action found for category: {category}")
            
            # Initialize execution state
            state = ExecutionState()
            
            # Main execution loop
            while state.rounds < self.max_rounds:
                try:
                    logger.info(f"Starting round {state.rounds + 1}/{self.max_rounds}")
                    
                    # Execute action
                    artifact = await action.run(ctx, self.toolset)
                    state.artifacts.append(artifact)
                    
                    # Update state from artifact metadata
                    if hasattr(artifact, 'meta'):
                        state.plan = getattr(artifact.meta, "plan", None) or state.plan
                        if hasattr(artifact.meta, "extracted"):
                            state.extracted = artifact.meta.extracted
                    
                    # Evaluate the artifact
                    report = await self.evaluator.evaluate(
                        category, 
                        state.plan,
                        artifact, 
                        state.extracted
                    )
                    
                    logger.info(f"Round {state.rounds + 1} evaluation: {'PASSED' if report.passed else 'FAILED'}")
                    
                    # Check if we're done
                    if report.passed:
                        logger.info("Execution completed successfully")
                        return artifact
                    
                    # Prepare for next round with remediation hints
                    if state.rounds < self.max_rounds - 1:
                        ctx.hints = self._merge_hints(ctx.hints, {
                            "remediation": report.suggestions,
                            "round": state.rounds + 1,
                            "previous_artifacts": state.artifacts,
                            "execution_state": state
                        })
                        logger.info(f"Preparing round {state.rounds + 2} with remediation hints")
                    
                    state.rounds += 1
                    
                except Exception as e:
                    logger.error(f"Error in round {state.rounds + 1}: {str(e)}")
                    state.rounds += 1
                    if state.rounds >= self.max_rounds:
                        raise
                    # Continue to next round with error context
                    ctx.hints = self._merge_hints(ctx.hints, {
                        "error": str(e),
                        "round": state.rounds
                    })
            
            # If we've exhausted all rounds, return the best artifact we have
            logger.warning(f"Max rounds ({self.max_rounds}) reached, returning best artifact")
            return state.artifacts[-1] if state.artifacts else None
            
        except Exception as e:
            logger.error(f"Fatal error in ChasorCore.run: {str(e)}")
            raise
    
    def _merge_hints(self, existing_hints: Optional[Dict[str, Any]], new_hints: Dict[str, Any]) -> Dict[str, Any]:
        """Safely merge hints dictionaries"""
        if existing_hints is None:
            return new_hints
        return {**existing_hints, **new_hints}
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all components.
        
        Returns:
            Dict with component health status
        """
        health = {}
        
        try:
            # Check router
            health['router'] = hasattr(self.router, 'classify') and callable(self.router.classify)
            
            # Check registry
            health['registry'] = hasattr(self.registry, 'get') and hasattr(self.registry, 'route')
            
            # Check toolset
            health['toolset'] = self.toolset is not None
            
            # Check evaluator
            health['evaluator'] = hasattr(self.evaluator, 'evaluate') and callable(self.evaluator.evaluate)
            
            health['overall'] = all(health.values())
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            health['overall'] = False
            health['error'] = str(e)
        
        return health
            
            
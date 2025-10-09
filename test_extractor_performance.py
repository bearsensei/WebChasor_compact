"""
Extractor Performance Test
Tests concurrent extraction performance vs sequential
"""

import os
import sys
import time
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[PERF-TEST] Loaded environment from {env_path}\n")
    else:
        print(f"[PERF-TEST] .env file not found at {env_path}\n")
except ImportError:
    print("[PERF-TEST] python-dotenv not installed, skipping .env file loading\n")

from actions.extractor import InformationExtractor, ExtractedVariable
from actions.ranker import ContentPassage
from planner import PlanTask, ExtractionPlan
from config_manager import get_config

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# Number of test tasks to simulate
NUM_TASKS = 10

# Simulated API delay (seconds) - mimics LLM API call latency
API_DELAY = 0.3

# Test with real LLM API or mock mode
USE_REAL_API = False  # Set to True to test with real OpenAI API

# ============================================================================
# Mock LLM Client for Testing
# ============================================================================

class MockLLMResponse:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.message = MockLLMResponse(content)

class MockLLMCompletion:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

class MockChatCompletion:
    """Mock OpenAI chat completion API"""
    
    async def create(self, **kwargs):
        # Simulate API latency
        await asyncio.sleep(API_DELAY)
        
        # Return mock response
        mock_response = {
            "value": f"Mock answer for {kwargs.get('messages', [{}])[1].get('content', '')[:50]}",
            "confidence": 0.85,
            "reasoning": "Mock extraction based on simulated passages",
            "source_quote": "This is a mock quote from the passage"
        }
        
        import json
        return MockLLMCompletion(json.dumps(mock_response))

class MockLLMClient:
    """Mock OpenAI client for testing"""
    def __init__(self):
        self.chat = type('obj', (object,), {'completions': MockChatCompletion()})()

# ============================================================================
# Test Data Generation
# ============================================================================

def generate_test_tasks(num_tasks: int) -> list:
    """Generate mock extraction tasks"""
    tasks = []
    for i in range(1, num_tasks + 1):
        task = PlanTask(
            variable_name=f"var_{i}",
            fact=f"What is the answer to question {i}?",
            category="fact",
            confidence_threshold=0.75
        )
        tasks.append(task)
    return tasks

def generate_test_passages(num_tasks: int) -> dict:
    """Generate mock passages for each task"""
    ranked_passages = {}
    
    for i in range(1, num_tasks + 1):
        var_name = f"var_{i}"
        
        # Create snippet passages (for Stage 1)
        snippet_passages = [
            ContentPassage(
                text=f"Snippet {j} for {var_name}: This contains relevant information.",
                source_url="SERP_SNIPPET",
                score=0.9 - j*0.1,
                metadata={
                    'snippet': f"Short snippet {j}",
                    'title': f"Result {j}",
                    'provenance': f"https://example.com/{var_name}/{j}"
                }
            )
            for j in range(1, 4)
        ]
        
        # Create deep passages (for Stage 2)
        deep_passages = [
            ContentPassage(
                text=f"Deep passage {j} for {var_name}: " + "This is longer content " * 20,
                source_url=f"https://example.com/{var_name}/page{j}",
                score=0.85 - j*0.05,
                metadata={
                    'title': f"Page {j}",
                    'provenance': f"https://example.com/{var_name}/page{j}"
                }
            )
            for j in range(1, 4)
        ]
        
        ranked_passages[var_name] = snippet_passages + deep_passages
    
    return ranked_passages

# ============================================================================
# Performance Test Functions
# ============================================================================

async def test_concurrent_extraction():
    """Test the optimized concurrent extraction"""
    print("\n" + "=" * 80)
    print("[PERF-TEST] Testing CONCURRENT Extraction (Optimized)")
    print("=" * 80)
    
    # Setup
    client = MockLLMClient() if not USE_REAL_API else None
    extractor = InformationExtractor(llm_client=client)
    
    tasks = generate_test_tasks(NUM_TASKS)
    ranked_passages = generate_test_passages(NUM_TASKS)
    plan = ExtractionPlan(tasks_to_extract=tasks)
    
    # Get configuration
    stage1_concurrency = extractor.stage1_concurrency
    stage2_concurrency = extractor.stage2_concurrency
    
    print(f"\n[PERF-TEST] Configuration:")
    print(f"  Tasks: {NUM_TASKS}")
    print(f"  Stage 1 Concurrency: {stage1_concurrency}")
    print(f"  Stage 2 Concurrency: {stage2_concurrency}")
    print(f"  Simulated API Delay: {API_DELAY}s")
    print(f"  Using Real API: {USE_REAL_API}")
    
    # Run extraction
    start_time = time.perf_counter()
    results = await extractor.extract_variables(plan, ranked_passages)
    elapsed_time = time.perf_counter() - start_time
    
    print(f"\n[PERF-TEST] Results:")
    print(f"  Extracted Variables: {len(results)}/{NUM_TASKS}")
    print(f"  Total Time: {elapsed_time:.2f}s")
    print(f"  Average per Task: {elapsed_time/NUM_TASKS:.3f}s")
    
    # Calculate theoretical improvement
    sequential_time = NUM_TASKS * API_DELAY * 2  # Stage 1 + Stage 2
    theoretical_concurrent = (NUM_TASKS / stage1_concurrency) * API_DELAY + \
                            (NUM_TASKS / stage2_concurrency) * API_DELAY
    
    print(f"\n[PERF-TEST] Performance Analysis:")
    print(f"  Sequential (theoretical): {sequential_time:.2f}s")
    print(f"  Concurrent (theoretical): {theoretical_concurrent:.2f}s")
    print(f"  Actual Time: {elapsed_time:.2f}s")
    print(f"  Speedup vs Sequential: {sequential_time/elapsed_time:.1f}x")
    
    return elapsed_time, len(results)

async def test_sequential_extraction_simulation():
    """Simulate sequential extraction for comparison"""
    print("\n" + "=" * 80)
    print("[PERF-TEST] Simulating SEQUENTIAL Extraction (for comparison)")
    print("=" * 80)
    
    tasks = generate_test_tasks(NUM_TASKS)
    
    print(f"\n[PERF-TEST] Configuration:")
    print(f"  Tasks: {NUM_TASKS}")
    print(f"  Simulated API Delay: {API_DELAY}s per task")
    
    start_time = time.perf_counter()
    
    # Simulate sequential processing
    for i, task in enumerate(tasks, 1):
        # Stage 1: Sequential
        await asyncio.sleep(API_DELAY)
        print(f"[PERF-TEST][Sequential] Task {i}/{NUM_TASKS} Stage1 completed")
        
        # Stage 2: Sequential (if needed)
        await asyncio.sleep(API_DELAY)
        print(f"[PERF-TEST][Sequential] Task {i}/{NUM_TASKS} Stage2 completed")
    
    elapsed_time = time.perf_counter() - start_time
    
    print(f"\n[PERF-TEST] Sequential Results:")
    print(f"  Total Time: {elapsed_time:.2f}s")
    print(f"  Average per Task: {elapsed_time/NUM_TASKS:.3f}s")
    
    return elapsed_time

# ============================================================================
# Main Test Runner
# ============================================================================

async def main():
    """Run performance comparison tests"""
    print("\n" + "=" * 80)
    print("[PERF-TEST] Extractor Performance Test")
    print("=" * 80)
    
    try:
        # Test 1: Sequential (simulated)
        sequential_time = await test_sequential_extraction_simulation()
        
        # Test 2: Concurrent (actual optimized code)
        concurrent_time, num_results = await test_concurrent_extraction()
        
        # Final comparison
        print("\n" + "=" * 80)
        print("[PERF-TEST] FINAL COMPARISON")
        print("=" * 80)
        print(f"\nSequential Time: {sequential_time:.2f}s")
        print(f"Concurrent Time: {concurrent_time:.2f}s")
        print(f"Speedup: {sequential_time/concurrent_time:.1f}x faster")
        print(f"Time Saved: {sequential_time - concurrent_time:.2f}s ({(1 - concurrent_time/sequential_time)*100:.1f}% reduction)")
        
        print("\n[PERF-TEST] Performance test completed successfully")
        
    except Exception as e:
        print(f"\n[PERF-TEST][ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


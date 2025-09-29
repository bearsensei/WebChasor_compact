#!/usr/bin/env python3
# test_thinking.py
"""
Test the fake thinking process generator in thinking_plan.py
"""
import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path - fix the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from thinking_plan import generate_fake_thinking


async def test_thinking():
    """Test thinking process generation"""
    print("=== Testing Fake Thinking Process Generator ===\n")
    
    test_query = "解释一下为什么会有潮汐？"
    print(f"Test Question: {test_query}\n")
    print("Generated Thinking Process (OpenAI SSE format):\n")
    
    await generate_fake_thinking(test_query)


if __name__ == "__main__":
    asyncio.run(test_thinking())
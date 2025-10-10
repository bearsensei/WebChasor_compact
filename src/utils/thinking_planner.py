# thinking_plan.py
"""
Simple fake thinking process generator
Generates detailed thinking process with <think></think> tags, supports OpenAI format streaming output
"""
import os
import sys

# Add src directory to path to find config_manager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_manager import get_config

class ThinkingPlan:
    """Generate fake thinking process"""
    
    # System prompt for thinking generation
    SYSTEM_PROMPT = "You are an AI assistant good at thinking. Please generate detailed, natural thinking processes to help analyze and solve problems."
    
    @staticmethod
    def build_thinking_prompt(query: str) -> str:
        """
        Build thinking prompt for a given query
        
        Args:
            query: User query
            
        Returns:
            Formatted thinking prompt
        """
        return f"""First, you are a helpful assistant, named HKChat 港话通. You need to check if the question need a thinking trace. If the quesiton belong to following categories: INFORMATION_RETRIEVAL, MATH_QUERY, TASK_PRODUCTIVITY , KNOWLEDGE_REASONING, CREATIVE_GENERATION, MULTIMODAL_QUERY,  please generate a detailed thinking trace for the following question. 
        If it like a conversational followup, greeting, general question, identity question, please also generate a relative comprehensive thinking process.

For detailed thinking trace, requirements:
1. Thinking trace should be in the same language as the question.
2. Think through steps to solve the problem comprehensively. BUT do not include the answer, just raise the questions and outline the approach.
3. Break down into multiple dimensions: information gathering, verification strategy, alternative approaches, potential challenges, quality check.
4. Use natural language with clear structure: First I need to confirm...Second, I need to think about...Then, I need to double check...Maybe I also need to... Lastly, I should finalize the answer...
5. Be thorough and detailed in your thinking process to demonstrate careful analysis.
6. Do not include <think></think> tags, I will add them automatically

Question: {query}

Please generate a detailed thinking process:"""
    
    def __init__(self):
        self.cfg = get_config()
    
    async def generate_thinking_stream(self, query: str, stop_event=None):
        """
        Generate thinking process streaming output (OpenAI SSE format)
        
        Args:
            query: User query
            stop_event: Optional asyncio.Event to signal early stop
        """
        import asyncio
        
        api_base = self.cfg.get('external_services.openai.api_base', 'https://api.openai.com/v1')
        api_key = os.getenv("OPENAI_API_KEY_AGENT")
        model = self.cfg.get('models.synthesizer.model_name', 'gpt-4')
        
        if not api_key:
            print("Error: No API key found")
            return
        
        # Build thinking prompt using static method
        thinking_prompt = self.build_thinking_prompt(query)
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=api_base)
            
            # Output opening tag first
            print("data: " + '{"choices":[{"delta":{"content":"<think>\\n"}}]}' + "\n\n", flush=True)
            
            # Create streaming request
            stream = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "system", 
                    "content": self.SYSTEM_PROMPT
                }, {
                    "role": "user", 
                    "content": thinking_prompt
                }],
                temperature=0.3,
                max_tokens=4000,
                stream=True
            )
            
            # Process streaming response
            for chunk in stream:
                # Check if external stop signal received
                if stop_event and stop_event.is_set():
                    break
                
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                    # Output original chunk JSON format
                    print(f"data: {chunk.model_dump_json()}\n\n", flush=True)
            
            # Output closing tag
            print("data: " + '{"choices":[{"delta":{"content":"\\n</think>\\n\\n"}}]}' + "\n\n", flush=True)
            
            # Send stream end signal
            print("data: [DONE]\n\n", flush=True)
            
        except Exception as e:
            print(f"Thinking generation error: {e}")


# Convenience function
async def generate_fake_thinking(query: str, stop_event=None):
    """Convenient fake thinking generation function"""
    thinking = ThinkingPlan()
    await thinking.generate_thinking_stream(query, stop_event)
# thinking_coordinator.py
"""
协调 thinking_plan 和 main 执行的模块
确保在同一个流式连接中先输出思考过程，再输出主要答案
"""
import asyncio
import os
from typing import Optional, AsyncGenerator
from config_manager import get_config


class ThinkingCoordinator:
    """协调思考过程和主要回答的统一流式输出"""
    
    def __init__(self):
        self.cfg = get_config()
    
    async def unified_stream_response(self, query: str, main_execution_func, *args, **kwargs):
        """
        统一的流式响应：先输出思考过程，再输出主要答案，最后发送 [DONE]
        
        Args:
            query: 用户查询
            main_execution_func: 主要执行函数
            *args, **kwargs: 传递给主要执行函数的参数
        """
        try:
            # 第一阶段：输出思考过程（不发送 [DONE]）
            await self._stream_thinking_process(query)
            
            # 输出分隔标记
            separator_content = "\\n\\n" + "="*50 + "\\n正在分析问题并生成答案...\\n" + "="*50 + "\\n\\n"
            separator_chunk = f'{{"choices":[{{"delta":{{"content":"{separator_content}"}}}}]}}'
            print(f"data: {separator_chunk}\\n\\n", flush=True)
            
            # 第二阶段：输出主要答案（不发送 [DONE]）
            await self._stream_main_response(query, main_execution_func, *args, **kwargs)
            
            # 最终发送 [DONE] 信号
            print("data: [DONE]\\n\\n", flush=True)
            
        except Exception as e:
            error_content = f"\\n\\n[ERROR] 处理过程中出现错误: {str(e)}\\n"
            error_chunk = f'{{"choices":[{{"delta":{{"content":"{error_content}"}}}}]}}'
            print(f"data: {error_chunk}\\n\\n", flush=True)
            print("data: [DONE]\\n\\n", flush=True)
    
    async def _stream_thinking_process(self, query: str):
        """输出思考过程（修改版，不发送 [DONE]）"""
        api_base = self.cfg.get('external_services.openai.api_base', 'https://api.openai.com/v1')
        api_key = os.getenv("OPENAI_API_KEY_AGENT")
        model = self.cfg.get('models.synthesizer.model_name', 'gpt-4')
        
        if not api_key:
            error_content = "Error: No API key found for thinking process"
            error_chunk = f'{{"choices":[{{"delta":{{"content":"{error_content}"}}}}]}}'
            print(f"data: {error_chunk}\\n\\n", flush=True)
            return
        
        thinking_prompt = f"""Please generate a detailed thinking process for the following question. Requirements:
1. Analyze key elements of the problem
2. Think through steps to solve the problem. BUT do not include the answer, just raise the questions.
3. List all possible angles and approaches
4. Use natural language and clear logic
5. Do not include <think></think> tags, I will add them automatically

Question: {query}

Please generate a detailed thinking process:"""
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=api_base)
            
            # 输出思考开始标签
            thinking_start = '<think>\\n'
            start_chunk = f'{{"choices":[{{"delta":{{"content":"{thinking_start}"}}}}]}}'
            print(f"data: {start_chunk}\\n\\n", flush=True)
            
            # 创建流式请求
            stream = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "system", 
                    "content": "You are an AI assistant good at thinking. Please generate detailed, natural thinking processes to help analyze and solve problems."
                }, {
                    "role": "user", 
                    "content": thinking_prompt
                }],
                temperature=0.3,
                max_tokens=1000,
                stream=True
            )
            
            # 处理流式响应（不发送 [DONE]）
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                    print(f"data: {chunk.model_dump_json()}\\n\\n", flush=True)
            
            # 输出思考结束标签
            thinking_end = '\\n</think>\\n'
            end_chunk = f'{{"choices":[{{"delta":{{"content":"{thinking_end}"}}}}]}}'
            print(f"data: {end_chunk}\\n\\n", flush=True)
            
        except Exception as e:
            error_content = f"Thinking generation error: {e}\\n"
            error_chunk = f'{{"choices":[{{"delta":{{"content":"{error_content}"}}}}]}}'
            print(f"data: {error_chunk}\\n\\n", flush=True)
    
    async def _stream_main_response(self, query: str, main_execution_func, *args, **kwargs):
        """输出主要响应（修改版，不发送 [DONE]）"""
        try:
            # 执行主要逻辑，但捕获其输出
            result = await main_execution_func(*args, **kwargs)
            
            if result:
                # 将结果作为流式输出发送
                content = str(result)
                # 将长文本分块发送
                chunk_size = 50  # 每块字符数
                for i in range(0, len(content), chunk_size):
                    chunk_content = content[i:i+chunk_size]
                    # 转义特殊字符
                    escaped_content = chunk_content.replace('\\', '\\\\').replace('"', '\\"').replace('\\n', '\\\\n')
                    response_chunk = f'{{"choices":[{{"delta":{{"content":"{escaped_content}"}}}}]}}'
                    print(f"data: {response_chunk}\\n\\n", flush=True)
                    await asyncio.sleep(0.05)  # 小延迟模拟流式效果
            
        except Exception as e:
            error_content = f"Main response error: {e}\\n"
            error_chunk = f'{{"choices":[{{"delta":{{"content":"{error_content}"}}}}]}}'
            print(f"data: {error_chunk}\\n\\n", flush=True)


# 全局协调器实例
thinking_coordinator = ThinkingCoordinator()


async def coordinate_unified_response(query: str, main_execution_func, *args, **kwargs):
    """统一协调函数"""
    await thinking_coordinator.unified_stream_response(query, main_execution_func, *args, **kwargs)
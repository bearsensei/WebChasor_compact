# streaming.py
"""
独立的流式输出模块
支持OpenAI格式的Server-Sent Events (SSE) 输出
"""
import os
import json
from typing import Optional, AsyncGenerator, Dict, Any
from config_manager import get_config


class StreamingOutput:
    """处理流式输出的类"""
    
    def __init__(self):
        self.cfg = get_config()
        self.enabled = self.cfg.get('demo.streaming.enabled', False)
        self.format = self.cfg.get('demo.streaming.format', 'openai')
    
    async def stream_openai_response(self, query: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        使用OpenAI API进行流式响应
        
        Args:
            query: 用户查询
            system_prompt: 系统提示词，如果为None则使用默认
            
        Returns:
            完整的响应内容，如果出错则返回None
        """
        api_base = self.cfg.get('external_services.openai.api_base', 'https://api.openai.com/v1')
        api_key = os.getenv("OPENAI_API_KEY_AGENT")
        model = self.cfg.get('models.synthesizer.model_name', 'gpt-4')
        
        if not api_key:
            print("Error: No API key found")
            return None
        
        # 默认系统提示词
        if system_prompt is None:
            system_prompt = "你是WebChasor，一个友好、专业的AI助手。请用中文回答问题，语调要自然、亲切。"
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=api_base)
            
            # 创建流式请求
            stream = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "system", 
                    "content": system_prompt
                }, {
                    "role": "user", 
                    "content": query
                }],
                temperature=0.1,
                max_tokens=2000,
                stream=True
            )
            
            # 处理流式响应
            full_content = ""
            
            if self.enabled and self.format == "openai":
                # OpenAI SSE格式输出
                for chunk in stream:
                    print(f"data: {chunk.model_dump_json()}\n\n", flush=True)
                    
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                        full_content += chunk.choices[0].delta.content
                
                # 发送流结束信号
                print("data: [DONE]\n\n", flush=True)
            else:
                # 简单文本输出（非流式或调试模式）
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end='', flush=True)
                        full_content += content
                print()  # 换行
            
            return full_content
            
        except Exception as e:
            print(f"Streaming error: {e}")
            return None
    
    def is_streaming_enabled(self) -> bool:
        """检查是否启用了流式输出"""
        return self.enabled
    
    def get_format(self) -> str:
        """获取输出格式"""
        return self.format
    
    def print_streaming_info(self):
        """打印流式输出配置信息"""
        print(f"[STREAMING][CONFIG] enabled={self.enabled}, format={self.format}")


# 全局实例
streaming_output = StreamingOutput()


# 便捷函数
async def stream_response(query: str, system_prompt: Optional[str] = None) -> Optional[str]:
    """便捷的流式响应函数"""
    return await streaming_output.stream_openai_response(query, system_prompt)


def is_streaming_enabled() -> bool:
    """检查是否启用流式输出"""
    return streaming_output.is_streaming_enabled()


def get_streaming_format() -> str:
    """获取流式输出格式"""
    return streaming_output.get_format() 
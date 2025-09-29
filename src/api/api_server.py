# api_server.py
"""
FastAPI 服务器，集成 WebChasor 的思考过程和主要回答功能
完全兼容 OpenAI API 格式，支持 Chatbox 等客户端
"""
import asyncio
import sys
import os
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
import json
import time
import uuid

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import WebChasor components
from artifacts import ActionRegistry, Context
from synthesizer import Synthesizer
from router import Router
from actions.productivity import PRODUCTIVITY
from actions.reasoning import REASONING
from actions.ir_rag import IR_RAG
from toolsets import Toolset
from config_manager import get_config


class Message(BaseModel):
    """消息模型"""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI 兼容的聊天完成请求"""
    model: str = "webchasor-thinking"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0


class WebChasorService:
    """WebChasor 服务封装"""
    
    def __init__(self):
        self.router = None
        self.registry = None
        self.toolset = None
        self.initialized = False
    
    async def initialize(self):
        """初始化 WebChasor 组件"""
        if self.initialized:
            return
        
        print("[SERVICE] Initializing WebChasor components...")
        
        # Load configuration
        cfg = get_config()
        print(f"[SERVICE] Configuration loaded")
        
        # Initialize components
        synthesizer = Synthesizer()
        self.router = Router()
        
        # Create toolset
        self.toolset = Toolset(
            router=self.router,
            synthesizer=synthesizer
        )
        
        # Initialize actions
        productivity_action = PRODUCTIVITY()
        reasoning_action = REASONING()
        
        # Create OpenAI client for IR_RAG if credentials are available
        llm_client = None
        api_base = cfg.get('external_services.openai.api_base')
        api_key = os.getenv("OPENAI_API_KEY_AGENT")
        
        if api_base and api_key:
            try:
                import openai
                llm_client = openai.OpenAI(api_key=api_key, base_url=api_base)
                print(f"[SERVICE] OpenAI client ready")
            except ImportError:
                print("[SERVICE] OpenAI library not available, using fallback")
        else:
            print("[SERVICE] No API credentials, using fallback")
        
        ir_rag_action = IR_RAG(llm_client=llm_client)
        
        # Create registry and register actions
        self.registry = ActionRegistry()
        self.registry._reg = {
            "PRODUCTIVITY": productivity_action,
            "REASONING": reasoning_action,
            "IR_RAG": ir_rag_action,
            "INFORMATION_RETRIEVAL": ir_rag_action,
        }
        
        self.initialized = True
        print("[SERVICE] WebChasor service initialized successfully")
    
    async def execute_query(self, query: str):
        """执行查询并返回结果"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # 路由查询
            category_enum = await self.router.classify("", query)
            category = category_enum.value
            
            # 获取action
            action_name = self.registry.route(category)
            action = self.registry.get(action_name)
            
            if not action:
                raise ValueError(f"No action found for category: {category}")
            
            # 创建Context并执行
            ctx = Context(
                history="", 
                query=query, 
                router_category=category, 
                hints={}
            )
            
            result = await action.run(ctx, self.toolset)
            return {
                "category": category,
                "action": action_name,
                "content": result.content,
                "meta": result.meta or {}
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


# 全局服务实例
webchasor_service = WebChasorService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化服务
    await webchasor_service.initialize()
    yield
    # 关闭时的清理工作（如果需要）


# 创建 FastAPI 应用
app = FastAPI(
    title="WebChasor API",
    description="WebChasor 智能问答服务，兼容 OpenAI API 格式",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_content_for_chatbox(content: str) -> str:
    """
    格式化内容以适配 Chatbox 显示
    处理换行符和特殊字符
    """
    # 替换 \n 为实际换行
    content = content.replace('\\n', '\n')
    
    # 确保引用格式正确显示
    content = content.replace('【', '[').replace('】', ']')
    
    return content


async def generate_thinking_stream_openai_format(query: str, request_model: str):
    """生成 OpenAI 格式的思考过程流式输出"""
    cfg = get_config()
    api_base = cfg.get('external_services.openai.api_base', 'https://api.openai.com/v1')
    api_key = os.getenv("OPENAI_API_KEY_AGENT")
    model = cfg.get('models.synthesizer.model_name', 'gpt-4')
    
    if not api_key:
        error_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request_model,
            "choices": [{
                "index": 0,
                "delta": {"content": "Error: No API key found"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        return
    
    thinking_prompt = f"""First, you are a helpful assistant. You need to check if the question need a thinking process. If it does, please generate a detailed thinking process for the following question. If it does not, please just finished the generation and output "no thinking process needed".Please generate a detailed thinking process for the following question. Requirements:
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
        start_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request_model,
            "choices": [{
                "index": 0,
                "delta": {"content": "<think>\n"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(start_chunk)}\n\n"
        
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
        
        # 处理流式响应，转换为 OpenAI 格式
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                openai_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk.choices[0].delta.content},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(openai_chunk)}\n\n"
        
        # 输出思考结束标签
        end_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request_model,
            "choices": [{
                "index": 0,
                "delta": {"content": "\n</think>\n\n"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(end_chunk)}\n\n"
        
    except Exception as e:
        error_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request_model,
            "choices": [{
                "index": 0,
                "delta": {"content": f"Thinking generation error: {str(e)}\n"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


async def generate_openai_compatible_stream(request: ChatCompletionRequest):
    """生成 OpenAI 兼容的流式响应 - 并行执行版本"""
    # 获取最后一条用户消息
    user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    if not user_message:
        error_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": "Error: No user message found"},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    try:
        # 立即启动主要查询任务（并行执行，不等待）
        print(f"[DEBUG] 启动主要查询任务...")
        main_task = asyncio.create_task(webchasor_service.execute_query(user_message))
        
        # 同时开始流式输出思考过程
        print(f"[DEBUG] 开始输出思考过程...")
        thinking_started = False
        async for chunk in generate_thinking_stream_openai_format(user_message, request.model):
            if not thinking_started:
                thinking_started = True
                print(f"[DEBUG] 思考过程开始输出，主要查询并行执行中...")
            yield chunk
        
        print(f"[DEBUG] 思考过程输出完成，等待主要查询结果...")
        
        # 思考过程输出完成后，等待主要查询结果
        result = await main_task
        content = format_content_for_chatbox(result["content"])
        
        print(f"[DEBUG] 主要查询完成，开始输出答案...")
        
        # 将内容分块流式输出
        chunk_size = 50
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i+chunk_size]
            
            # 生成 OpenAI 格式的响应
            chunk_data = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": chunk_content
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)  # 减少延迟，加快输出
        
        # 发送结束信号
        final_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk", 
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": f"\n\n[ERROR] 处理过程中出现错误: {str(e)}\n"
                },
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    OpenAI 兼容的聊天完成接口
    完全兼容 Chatbox 等客户端
    """
    # 简单的 API key 验证（可选）
    # if authorization and not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    if request.stream:
        # 流式响应
        return StreamingResponse(
            generate_openai_compatible_stream(request),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
    else:
        # 非流式响应
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        result = await webchasor_service.execute_query(user_message)
        formatted_content = format_content_for_chatbox(result["content"])
        
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": formatted_content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(formatted_content.split()),
                "total_tokens": len(user_message.split()) + len(formatted_content.split())
            }
        }


@app.get("/v1/models")
async def list_models():
    """列出可用模型（Chatbox 需要）"""
    return {
        "object": "list",
        "data": [
            {
                "id": "webchasor-thinking",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "webchasor",
                "permission": [],
                "root": "webchasor-thinking",
                "parent": None
            }
        ]
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "WebChasor API",
        "initialized": webchasor_service.initialized
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "WebChasor API Server - OpenAI Compatible",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
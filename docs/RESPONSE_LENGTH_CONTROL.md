# Response Length Control Implementation

## 📋 概述

本文档说明如何通过 `config.yaml` 统一控制不同 Router 类别的回答长度和温度参数。

---

## ✅ 已完成的修改

### 1. **config.yaml 配置添加** ✅

在 `config/config.yaml` 中添加了新的 `response_length` 配置节：

```yaml
# Response length control by category
response_length:
  # Information Retrieval responses
  information_retrieval:
    max_tokens: 8000          # 综合分析类问题（超长回答）
    temperature: 0.8
    
  # Knowledge Reasoning responses  
  knowledge_reasoning:
    max_tokens: 6000          # 推理分析类问题（长回答）
    temperature: 0.7
    
  # Geographic Query responses
  geo_query:
    max_tokens: 500           # 地理查询（简洁）
    temperature: 0.3
    
  # Conversational Followup responses
  conversational_followup:
    max_tokens: 300           # 简单追问（简短）
    temperature: 0.6
    
  # Task Productivity responses
  task_productivity:
    max_tokens: 2000          # 生产力任务
    temperature: 0.0
    
  # Default fallback
  default:
    max_tokens: 3000
    temperature: 0.7
```

### 2. **config_manager.py 辅助方法添加** ✅

在 `src/config_manager.py` 中添加了 `get_response_length_config()` 方法：

```python
def get_response_length_config(self, category: str) -> Dict[str, Any]:
    """
    Get response length configuration for a specific category.
    
    Args:
        category: Router category (e.g., 'INFORMATION_RETRIEVAL', 'GEO_QUERY')
    
    Returns:
        Dict with max_tokens and temperature
    """
    # Normalize category name to config key
    category_key = category.lower()
    
    # Get category-specific config
    config_path = f'response_length.{category_key}'
    category_config = self.get(config_path, {})
    
    # Fallback to default if not found
    if not category_config:
        category_config = self.get('response_length.default', {
            'max_tokens': 3000,
            'temperature': 0.7
        })
    
    return category_config
```

---

## 🔧 待完成的修改

### 3. **synthesizer.py 修改** ⏳

需要修改 `src/synthesizer.py` 的以下部分：

#### 3.1 修改 `_create_default_llm()` 方法

```python
def _create_default_llm(self):
    """Create default LLM from environment variables"""
    api_base = get_config().get('external_services.openai.api_base', 'https://api.openai.com/v1')
    api_key = os.getenv("OPENAI_API_KEY_AGENT") 
    model = get_config().get('models.synthesizer.model_name', 'gpt-4')
    
    if not api_base or not api_key:
        print("[SYNTHESIZER][WARN] No API credentials, using mock LLM")
        return self._mock_llm
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=api_base)
        
        # 修改：支持动态 max_tokens
        async def real_llm(prompt, temperature=0, max_tokens=None):
            # 如果没有指定 max_tokens，使用 config 中的默认值
            if max_tokens is None:
                max_tokens = get_config().get('models.synthesizer.max_tokens', 2000)
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens  # 使用传入的 max_tokens
            )
            return response.choices[0].message.content
        
        return real_llm
        
    except ImportError:
        print("⚠️ OpenAI library not available, using mock LLM")
        return self._mock_llm
    except Exception as e:
        print(f"⚠️ Failed to initialize OpenAI client: {e}, using mock LLM")
        return self._mock_llm
```

#### 3.2 修改 `generate()` 方法

```python
async def generate(self, category, style_key, constraints, materials, task_scaffold=None):
    # 1. 自动语言检测
    auto_lang = detect_lang_4way(materials)
    print(f'[SYNTHESIZER][LANG_DEBUG] auto_lang: {auto_lang}')
    
    # 2. 从 config 获取该类别的长度配置
    cfg = get_config()
    length_config = cfg.get_response_length_config(category)
    
    # 3. 使用 constraints 中的值，如果没有则使用 config 中的值
    max_tokens = constraints.get('max_tokens', length_config.get('max_tokens', 3000))
    temperature = constraints.get('temperature', length_config.get('temperature', self.temperature))
    
    # 4. 构建 prompt（添加长度提示）
    instruction_hint = constraints.get('instruction_hint', '')
    
    # 根据 max_tokens 添加长度提示
    if max_tokens <= 500:
        length_hint = f"\n\nIMPORTANT: Keep response concise and focused (approximately {max_tokens} tokens / {int(max_tokens * 0.75)} words)."
    elif max_tokens <= 2000:
        length_hint = f"\n\nIMPORTANT: Provide a well-structured response (approximately {max_tokens} tokens / {int(max_tokens * 0.75)} words)."
    else:
        length_hint = f"\n\nIMPORTANT: Provide a comprehensive response (approximately {max_tokens} tokens / {int(max_tokens * 0.75)} words)."
    
    instruction_hint = instruction_hint + length_hint
    
    # 5. 渲染 prompt
    prompt = render_synthesizer_prompt(
        action_policy=SYNTHESIZER_ACTION_POLICIES.get(category, "Provide helpful and accurate responses."),
        materials=materials,
        user_query=materials,
        language=auto_lang,
        reading_level=constraints.get("reading_level", "general"),
        preferred_style=(style_key if style_key and style_key != "auto" else None),
        global_system=SYNTHESIZER_GLOBAL_SYSTEM,
        internal_scaffold=(SYNTHESIZER_HIDDEN_REASONING_SCAFFOLD if category == "KNOWLEDGE_REASONING" else ""),
        instruction_hint=instruction_hint
    )

    print(f"[SYNTHESIZER][EXEC] model={self.model_name} temp={temperature} category={category} lang={auto_lang} max_tokens={max_tokens}")

    # 6. 调用 LLM（传入 max_tokens）
    response = await self.llm(prompt, temperature=temperature, max_tokens=max_tokens)
    return response
```

### 4. **prompt.py 添加提示词管理** ⏳

在 `src/prompt.py` 文件末尾添加：

```python
# ============================================================================
# ACTION-SPECIFIC INSTRUCTION HINTS
# ============================================================================

# IR_RAG instruction hints
IR_RAG_INSTRUCTION_HINTS = {
    "default": """Provide a comprehensive answer with clear sections. Use diverse formatting: bullet points, numbered lists, and tables where appropriate. Mix paragraphs with structured formats for better readability. DO NOT use citation numbers [1], [2], etc. in the main text. Only include a reference list at the end.""",
    
    "simple": """Provide a clear, well-structured answer (800-1000 words) with 3-4 key sections. Use bullet points and tables where helpful.""",
    
    "moderate": """Provide a detailed answer (2000-3000 words) with 5-6 sections. Mix paragraphs with lists and tables for better readability.""",
    
    "complex": """Provide a comprehensive analysis (6000-8000 words) with 8-10 detailed sections. Use diverse formatting: paragraphs, bullet points, numbered lists, comparison tables, and structured data presentations."""
}

# REASONING instruction hints
REASONING_INSTRUCTION_HINTS = {
    "analytical": """Focus on ANALYTICAL reasoning approach. Use diverse formatting: bullet points, numbered lists, comparison tables, and structured presentations where appropriate. Mix paragraphs with lists and tables for better readability.""",
    
    "comparative": """Focus on COMPARATIVE reasoning approach. Use comparison tables, pros/cons lists, and side-by-side analysis. Highlight key differences and similarities.""",
    
    "explanatory": """Focus on EXPLANATORY reasoning approach. Break down complex concepts into clear steps. Use diagrams descriptions, examples, and analogies where helpful.""",
    
    "predictive": """Focus on PREDICTIVE reasoning approach. Analyze trends, present scenarios, and discuss probabilities. Use data visualization descriptions where helpful.""",
}

# GEO_QUERY instruction hint
GEO_QUERY_INSTRUCTION_HINT = """User asked: "{user_query}"

Please rewrite the route information above to be more friendly and conversational while:
- Keeping the SAME language as the user's query
- Preserving ALL factual details (station names, distances, durations, transit lines)
- Making it easy to read and follow
- Adding a brief friendly greeting/closing if appropriate
- Keep it concise (150-200 words)"""

# Helper function
def get_length_hint(max_tokens: int) -> str:
    """Generate appropriate length hint based on max_tokens"""
    if max_tokens <= 500:
        words = int(max_tokens * 0.75)
        return f"\n\nIMPORTANT: Keep response concise and focused (approximately {max_tokens} tokens / {words} words)."
    elif max_tokens <= 2000:
        words = int(max_tokens * 0.75)
        return f"\n\nIMPORTANT: Provide a well-structured response (approximately {max_tokens} tokens / {words} words)."
    else:
        words = int(max_tokens * 0.75)
        return f"\n\nIMPORTANT: Provide a comprehensive response (approximately {max_tokens} tokens / {words} words)."

def build_geo_query_instruction(user_query: str) -> str:
    """Build GEO_QUERY instruction hint"""
    return GEO_QUERY_INSTRUCTION_HINT.format(user_query=user_query)
```

### 5. **ir_rag.py 修改** ⏳

在 `src/actions/ir_rag.py` 的 `_synthesize_response()` 方法中：

```python
# 在文件开头添加导入
from prompt import IR_RAG_INSTRUCTION_HINTS, get_length_hint

# 在 _synthesize_response() 中修改
async def _synthesize_response(self, ctx: Context, plan: ExtractionPlan, 
                             extracted_vars: Dict[str, ExtractedVariable], 
                             search_results: List[SearchResult], toolset) -> Artifact:
    
    # 获取配置
    cfg = get_config()
    length_config = cfg.get_response_length_config("INFORMATION_RETRIEVAL")
    max_tokens = length_config.get('max_tokens', 8000)
    
    # 使用统一的 instruction hint
    instruction_hint = IR_RAG_INSTRUCTION_HINTS["default"]
    instruction_hint += get_length_hint(max_tokens)
    
    constraints = {
        "language": "auto",
        "tone": "factual, authoritative",
        "temperature": length_config.get('temperature', 0.8),
        "max_tokens": max_tokens,
        "instruction_hint": instruction_hint
    }
    
    response = await toolset.synthesizer.generate(
        category="INFORMATION_RETRIEVAL",
        style_key="auto",
        constraints=constraints,
        materials=f"Query: {ctx.query}\n\nExtracted Information:\n{materials}",
        task_scaffold=None
    )
    
    # ... 其余代码
```

### 6. **reasoning.py 修改** ⏳

在 `src/actions/reasoning.py` 的 `run()` 方法中：

```python
# 在文件开头添加导入
from prompt import REASONING_INSTRUCTION_HINTS, get_length_hint

# 在 run() 中修改
async def run(self, ctx: Context, toolset) -> Artifact:
    # ... 前面的代码
    
    # 获取配置
    cfg = get_config()
    length_config = cfg.get_response_length_config("KNOWLEDGE_REASONING")
    max_tokens = length_config.get('max_tokens', 6000)
    
    # 选择 instruction hint
    reasoning_key = reasoning_type.value.lower()
    instruction_hint = REASONING_INSTRUCTION_HINTS.get(reasoning_key, REASONING_INSTRUCTION_HINTS["analytical"])
    instruction_hint += get_length_hint(max_tokens)
    
    constraints = {
        "language": "auto",
        "tone": "friendly, conversational",
        "temperature": length_config.get('temperature', 0.7),
        "max_tokens": max_tokens,
        "instruction_hint": instruction_hint
    }
    
    text = await toolset.synthesizer.generate(
        category="KNOWLEDGE_REASONING",
        style_key="auto",
        constraints=constraints,
        materials=ctx.query,
        task_scaffold=None
    )
    
    # ... 其余代码
```

### 7. **geo_query.py 修改** ⏳

在 `src/actions/geo_query.py` 的 `_enhance_response()` 方法中：

```python
# 在文件开头添加导入
from prompt import build_geo_query_instruction, get_length_hint

# 在 _enhance_response() 中修改
async def _enhance_response(self, result: Artifact, ctx: Context) -> Artifact:
    print(f"[GEO_QUERY][ENHANCE] Starting enhancement...")
    
    try:
        # 获取配置
        cfg = get_config()
        length_config = cfg.get_response_length_config("GEO_QUERY")
        max_tokens = length_config.get('max_tokens', 500)
        
        # 使用统一的 instruction hint
        instruction_hint = build_geo_query_instruction(ctx.query)
        instruction_hint += get_length_hint(max_tokens)
        
        # 使用 synthesizer.generate() 而不是直接调用 llm()
        enhanced_content = await self.synthesizer.generate(
            category="GEO_QUERY",
            style_key="auto",
            constraints={
                "language": "auto",
                "tone": "friendly, conversational",
                "temperature": length_config.get('temperature', 0.3),
                "max_tokens": max_tokens,
                "instruction_hint": instruction_hint
            },
            materials=f"# Original Route Information\n\n{result.content}",
            task_scaffold=None
        )
        
        # 创建新的 artifact
        enhanced_result = Artifact(
            kind=result.kind,
            content=enhanced_content,
            meta={
                **result.meta,
                "enhanced": True,
                "original_length": len(result.content),
                "enhanced_length": len(enhanced_content),
                "max_tokens": max_tokens
            }
        )
        
        return enhanced_result
        
    except Exception as e:
        # ... 错误处理
```

---

## 📊 预期效果

修改完成后，不同类型的查询将有不同的回答长度：

| Router Category | max_tokens | 预期字数 | 用途 |
|----------------|------------|---------|------|
| **INFORMATION_RETRIEVAL** | 8000 | ~6000 字 | 综合分析类问题 |
| **KNOWLEDGE_REASONING** | 6000 | ~4500 字 | 推理分析类问题 |
| **GEO_QUERY** | 500 | ~375 字 | 地理查询（简洁） |
| **CONVERSATIONAL_FOLLOWUP** | 300 | ~225 字 | 简单追问 |
| **TASK_PRODUCTIVITY** | 2000 | ~1500 字 | 生产力任务 |

---

## 🧪 测试方法

修改完成后，可以通过以下方式测试：

```bash
# 测试地理查询（应该简短）
python src/main.py
# 输入: "从中环到尖沙咀怎么走？"
# 预期: ~500 tokens 的简洁回答

# 测试信息检索（应该详细）
# 输入: "香港今年中秋活动"
# 预期: ~8000 tokens 的详细回答

# 测试推理（应该中等长度）
# 输入: "为什么香港房价这么高？"
# 预期: ~6000 tokens 的分析回答
```

---

## 🔄 后续优化

1. **动态复杂度判断**: 根据查询复杂度自动选择 simple/moderate/complex
2. **用户偏好**: 允许用户在查询中指定回答长度（"简短回答"/"详细回答"）
3. **A/B 测试**: 测试不同长度对用户满意度的影响
4. **监控统计**: 记录实际生成的 token 数与配置的差异

---

## 📝 注意事项

1. **max_tokens 是上限**: LLM 可能生成更短的回答，但不会超过这个限制
2. **温度参数**: 较低的温度（0.3）适合事实性回答，较高的温度（0.8）适合创造性回答
3. **配置优先级**: `constraints` 参数 > `config.yaml` > 代码默认值
4. **向后兼容**: 如果 `response_length` 配置不存在，会使用默认值

---

最后更新: 2025-10-06

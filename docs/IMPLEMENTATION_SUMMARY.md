# 响应长度控制实施总结

## ✅ 已完成的修改

### 1. **config.yaml 配置** ✅
- 文件: `config/config.yaml`
- 添加了 `response_length` 配置节
- 为 6 种类别配置了 `max_tokens` 和 `temperature`

```yaml
response_length:
  information_retrieval:
    max_tokens: 8000
    temperature: 0.8
  knowledge_reasoning:
    max_tokens: 6000
    temperature: 0.7
  geo_query:
    max_tokens: 500      # ✅ GEO_QUERY 现在限制为 500 tokens
    temperature: 0.3
  conversational_followup:
    max_tokens: 300      # ✅ 追问限制为 300 tokens
    temperature: 0.6
  task_productivity:
    max_tokens: 2000
    temperature: 0.0
  default:
    max_tokens: 3000
    temperature: 0.7
```

### 2. **config_manager.py 辅助方法** ✅
- 文件: `src/config_manager.py`
- 添加了 `get_response_length_config()` 方法
- 支持按类别获取配置，带默认值回退

```python
def get_response_length_config(self, category: str) -> Dict[str, Any]:
    """Get response length configuration for a specific category"""
    category_key = category.lower()
    config_path = f'response_length.{category_key}'
    category_config = self.get(config_path, {})
    
    if not category_config:
        category_config = self.get('response_length.default', {
            'max_tokens': 3000,
            'temperature': 0.7
        })
    
    return category_config
```

### 3. **synthesizer.py 核心功能** ✅
- 文件: `src/synthesizer.py`
- 修改了 3 个关键方法

#### 3.1 `_create_default_llm()` - 支持动态 max_tokens
```python
async def real_llm(prompt, temperature=0, max_tokens=None):
    if max_tokens is None:
        max_tokens = get_config().get('models.synthesizer.max_tokens', 2000)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens  # ✅ 使用动态 max_tokens
    )
    return response.choices[0].message.content
```

#### 3.2 `generate()` - 从 config 读取长度配置
```python
async def generate(self, category, style_key, constraints, materials, task_scaffold=None):
    # 1. 自动语言检测
    auto_lang = detect_lang_4way(materials)
    
    # 2. 从 config 获取该类别的长度配置
    cfg = get_config()
    length_config = cfg.get_response_length_config(category)
    
    # 3. 使用 constraints 中的值，如果没有则使用 config 中的值
    max_tokens = constraints.get('max_tokens', length_config.get('max_tokens', 3000))
    temperature = constraints.get('temperature', length_config.get('temperature', self.temperature))
    
    # 4. 添加长度提示
    instruction_hint = constraints.get('instruction_hint', '')
    if max_tokens <= 500:
        length_hint = f"\n\nIMPORTANT: Keep response concise and focused (approximately {max_tokens} tokens / {int(max_tokens * 0.75)} words)."
    # ...
    
    # 5. 调用 LLM
    response = await self.llm(prompt, temperature=temperature, max_tokens=max_tokens)
    return response
```

#### 3.3 `_mock_llm()` - 支持 max_tokens 参数
```python
async def _mock_llm(self, prompt, temperature=0, max_tokens=None):
    # 测试时也支持 max_tokens 参数
    ...
```

### 4. **prompt.py 统一提示词管理** ✅
- 文件: `src/prompt.py`
- 添加了提示词字典和辅助函数

#### 4.1 IR_RAG 提示词
```python
IR_RAG_INSTRUCTION_HINTS = {
    "default": """Provide a comprehensive answer with clear sections...""",
    "simple": """Provide a clear, well-structured answer (800-1000 words)...""",
    "moderate": """Provide a detailed answer (2000-3000 words)...""",
    "complex": """Provide a comprehensive analysis (6000-8000 words)..."""
}
```

#### 4.2 REASONING 提示词
```python
REASONING_INSTRUCTION_HINTS = {
    "analytical": """Focus on ANALYTICAL reasoning approach...""",
    "comparative": """Focus on COMPARATIVE reasoning approach...""",
    "explanatory": """Focus on EXPLANATORY reasoning approach...""",
    "predictive": """Focus on PREDICTIVE reasoning approach...""",
    "problem_solving": """Focus on PROBLEM_SOLVING approach..."""
}
```

#### 4.3 GEO_QUERY 提示词模板
```python
GEO_QUERY_INSTRUCTION_HINT = """User asked: "{user_query}"

Please rewrite the route information above to be more friendly and conversational while:
- Keeping the SAME language as the user's query
- Preserving ALL factual details (station names, distances, durations, transit lines)
- Making it easy to read and follow
- Adding a brief friendly greeting/closing if appropriate
- Keep it concise (150-200 words)"""
```

#### 4.4 辅助函数
```python
def get_length_hint(max_tokens: int) -> str:
    """Generate appropriate length hint based on max_tokens"""
    if max_tokens <= 500:
        return f"\n\nIMPORTANT: Keep response concise and focused..."
    # ...

def build_geo_query_instruction(user_query: str) -> str:
    """Build GEO_QUERY instruction hint with user query"""
    return GEO_QUERY_INSTRUCTION_HINT.format(user_query=user_query)
```

### 5. **geo_query.py 集成** ✅
- 文件: `src/actions/geo_query.py`
- 修改了 `_enhance_response()` 方法

```python
async def _enhance_response(self, result: Artifact, ctx: Context) -> Artifact:
    try:
        from prompt import build_geo_query_instruction, get_length_hint
        from config_manager import get_config
        
        # 从 config 获取长度配置
        cfg = get_config()
        length_config = cfg.get_response_length_config("GEO_QUERY")
        max_tokens = length_config.get('max_tokens', 500)
        temperature = length_config.get('temperature', 0.3)
        
        # 使用统一的提示词构建器
        instruction_hint = build_geo_query_instruction(ctx.query)
        instruction_hint += get_length_hint(max_tokens)
        
        # 使用 synthesizer.generate() 统一接口
        enhanced_content = await self.synthesizer.generate(
            category="GEO_QUERY",
            style_key="auto",
            constraints={
                "language": "auto",
                "tone": "friendly, conversational",
                "temperature": temperature,
                "max_tokens": max_tokens,  # ✅ 限制为 500 tokens
                "instruction_hint": instruction_hint
            },
            materials=materials,
            task_scaffold=None
        )
        
        print(f"[GEO_QUERY][ENHANCE] Completed (max_tokens={max_tokens})")
        
        # 在 meta 中记录配置信息
        enhanced_result = Artifact(
            kind=result.kind,
            content=enhanced_content,
            meta={
                **result.meta,
                "enhanced": True,
                "original_length": len(result.content),
                "enhanced_length": len(enhanced_content),
                "max_tokens": max_tokens,  # ✅ 记录使用的 max_tokens
                "temperature": temperature
            }
        )
        
        return enhanced_result
        
    except Exception as e:
        # 错误处理...
```

---

## 🎯 实现效果

### GEO_QUERY 字数控制
- **配置**: `max_tokens: 500`
- **预期字数**: ~375 字
- **温度**: 0.3（更确定性，更简洁）
- **提示词**: 明确要求 "Keep it concise (150-200 words)"

### 其他类别
| 类别 | max_tokens | 预期字数 | 用途 |
|------|------------|---------|------|
| INFORMATION_RETRIEVAL | 8000 | ~6000 字 | 综合分析 |
| KNOWLEDGE_REASONING | 6000 | ~4500 字 | 推理分析 |
| **GEO_QUERY** | **500** | **~375 字** | **地理查询（简洁）** |
| CONVERSATIONAL_FOLLOWUP | 300 | ~225 字 | 简单追问 |
| TASK_PRODUCTIVITY | 2000 | ~1500 字 | 生产力任务 |

---

## 📋 配置优先级

系统使用三级配置优先级：

1. **constraints 参数** (最高优先级)
   ```python
   constraints = {"max_tokens": 1000}  # 覆盖 config
   ```

2. **config.yaml 配置** (中等优先级)
   ```yaml
   response_length:
     geo_query:
       max_tokens: 500
   ```

3. **代码默认值** (最低优先级)
   ```python
   max_tokens = 3000  # 如果前两者都没有
   ```

---

## 🧪 测试方法

### 测试 GEO_QUERY 字数限制

```bash
# 激活环境
conda activate webchaser

# 运行测试
python src/main.py
```

测试查询：
```
从中环到尖沙咀怎么走？
```

预期结果：
- 回答长度约 375 字（500 tokens）
- 包含所有关键信息（站名、距离、时长）
- 语气友好简洁
- 有问候和结尾

查看日志确认：
```
[GEO_QUERY][ENHANCE] Completed (max_tokens=500)
[SYNTHESIZER][EXEC] ... max_tokens=500
```

---

## ⏳ 待完成的工作

### 1. **ir_rag.py 修改** (可选)
如果需要为 IR_RAG 也添加字数控制：

```python
# 在 _synthesize_response() 中
cfg = get_config()
length_config = cfg.get_response_length_config("INFORMATION_RETRIEVAL")
max_tokens = length_config.get('max_tokens', 8000)

constraints = {
    "max_tokens": max_tokens,
    "temperature": length_config.get('temperature', 0.8),
    "instruction_hint": IR_RAG_INSTRUCTION_HINTS["default"] + get_length_hint(max_tokens)
}
```

### 2. **reasoning.py 修改** (可选)
如果需要为 REASONING 也添加字数控制：

```python
# 在 run() 中
cfg = get_config()
length_config = cfg.get_response_length_config("KNOWLEDGE_REASONING")
max_tokens = length_config.get('max_tokens', 6000)

constraints = {
    "max_tokens": max_tokens,
    "temperature": length_config.get('temperature', 0.7),
    "instruction_hint": REASONING_INSTRUCTION_HINTS[reasoning_key] + get_length_hint(max_tokens)
}
```

---

## 📝 使用说明

### 修改 GEO_QUERY 字数限制

只需修改 `config/config.yaml`：

```yaml
response_length:
  geo_query:
    max_tokens: 300      # 改为更短
    temperature: 0.2     # 改为更确定
```

无需修改任何代码！

### 为新类别添加字数控制

1. 在 `config.yaml` 中添加配置：
```yaml
response_length:
  my_new_category:
    max_tokens: 1000
    temperature: 0.5
```

2. 在 action 中使用：
```python
cfg = get_config()
length_config = cfg.get_response_length_config("MY_NEW_CATEGORY")
max_tokens = length_config.get('max_tokens', 3000)
```

---

## 🎉 总结

### 已实现的核心功能
1. ✅ 配置文件统一管理响应长度
2. ✅ Synthesizer 支持动态 max_tokens
3. ✅ 提示词统一管理在 prompt.py
4. ✅ GEO_QUERY 成功集成字数控制
5. ✅ 完整的配置优先级机制

### 关键改进
- **集中配置**: 所有长度控制在 config.yaml
- **统一接口**: 所有 action 使用 synthesizer.generate()
- **提示词管理**: 所有提示词在 prompt.py
- **灵活扩展**: 易于添加新类别
- **向后兼容**: 保留 constraints 参数覆盖

### GEO_QUERY 效果
- ✅ 从之前可能的长回答限制到 ~375 字
- ✅ 保持所有关键信息
- ✅ 更简洁友好的表达
- ✅ 可通过配置文件轻松调整

---

最后更新: 2025-10-06
作者: AI Assistant
状态: ✅ 核心功能已完成并可测试

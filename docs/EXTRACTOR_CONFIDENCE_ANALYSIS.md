# Extractor Confidence 0.5 问题分析

## 终端日志分析

```
🔍 EXTRACTOR: Extracted 'guoyi_profile' with confidence 0.50
🔍 EXTRACTOR: Extracted 'guoyi_current_position' with confidence 0.50
Extraction failed for guoyi_qualifications: 'NoneType' object has no attribute 'strip'
🔍 EXTRACTOR: Extracted 'guoyi_qualifications' with confidence 0.60
🔍 EXTRACTOR: Extracted 'guoyi_interest_statement' with confidence 0.50
🔍 EXTRACTOR: Extracted 'current_president_name' with confidence 0.50
Extraction failed for current_president_start: 'NoneType' object has no attribute 'strip'
🔍 EXTRACTOR: Extracted 'current_president_start' with confidence 0.60
[EXTRACTOR][DEBUG] Extracted value: None
[EXTRACTOR][DEBUG] Confidence: 0.0
[EXTRACTOR][DEBUG] Reasoning: None of the provided passages mention Hong Kong Polytechnic University or its current president, so the requested information cannot be extracted.
🔍 EXTRACTOR: Extracted 'current_president_end' with confidence 0.00
```

---

## 问题 1: Confidence 0.5 和 0.6 的含义

### Confidence 0.5 的来源

查看代码 `src/actions/ir_rag.py` 第 685-694 行：

```python
except json.JSONDecodeError:
    # Fallback if JSON parsing fails
    return ExtractedVariable(
        variable_name=task.variable_name,
        value=result_text,
        confidence=0.5,  # ⚠️ 硬编码的 0.5
        provenance=[p.source_url for p in passages],
        extraction_method="llm_fallback",
        raw_passages=[p.text for p in passages]
    )
```

**含义**: 
- LLM 返回的不是有效的 JSON
- 系统使用 fallback 机制
- **硬编码 confidence 为 0.5**

### Confidence 0.6 的来源

查看代码 `src/actions/ir_rag.py` 第 696-726 行：

```python
def _fallback_extract(self, task: PlanTask, passages: List[ContentPassage]) -> ExtractedVariable:
    """Fallback extraction using simple heuristics"""
    if not passages:
        return ExtractedVariable(
            variable_name=task.variable_name,
            value=None,
            confidence=0.0,
            extraction_method="fallback"
        )
    
    # Use the highest-scored passage as the answer
    best_passage = passages[0]
    
    # Simple extraction based on task category
    if task.category == "biography":
        value = self._extract_biographical_info(best_passage.text)
    elif task.category == "fact_verification":
        value = self._extract_factual_claim(best_passage.text)
    else:
        # Generic extraction - use first sentence or paragraph
        sentences = best_passage.text.split('.')
        value = sentences[0].strip() if sentences else best_passage.text[:200]
    
    return ExtractedVariable(
        variable_name=task.variable_name,
        value=value,
        confidence=0.6,  # ⚠️ 硬编码的 0.6
        provenance=[p.source_url for p in passages],
        extraction_method="fallback",
        raw_passages=[p.text for p in passages]
    )
```

**含义**:
- LLM extraction 失败（抛出异常）
- 使用简单的启发式方法提取
- 从第一个 passage 提取第一句话
- **硬编码 confidence 为 0.6**

---

## 问题 2: 为什么会触发 Fallback？

### 原因 A: JSON 解析失败 (confidence 0.5)

**可能的情况**:

1. **LLM 返回格式不正确**
   ```
   期望: {"value": "...", "confidence": 0.8, ...}
   实际: The extracted information is...
   ```

2. **LLM 返回包含额外文本**
   ```
   期望: {"value": "...", "confidence": 0.8, ...}
   实际: Here is the extracted information:
         {"value": "...", "confidence": 0.8, ...}
   ```

3. **LLM 返回不完整的 JSON**
   ```
   期望: {"value": "...", "confidence": 0.8, "reasoning": "..."}
   实际: {"value": "...", "confidence": 0.8  (truncated)
   ```

### 原因 B: LLM 调用异常 (confidence 0.6)

**可能的情况**:

1. **`'NoneType' object has no attribute 'strip'` 错误**
   ```python
   result_text = response.choices[0].message.content.strip()
   ```
   如果 `content` 是 `None`，就会报这个错误

2. **LLM 返回空响应**
   - `response.choices[0].message.content` 是 `None`
   - 可能是 API 问题或 token 限制

3. **其他异常**
   - 网络超时
   - API 限流
   - 模型错误

---

## 问题 3: 中文还是英文抽取？

### 查看 Extraction Prompt

从代码第 598-652 行可以看到 prompt：

```python
prompt = f"""
Extract the following information from the provided text passages:

QUESTION: {task.fact}
VARIABLE: {task.variable_name}
CATEGORY: {task.category}
CORE ENTITIES: {entities_str}

TEXT PASSAGES:
{combined_text}

**CRITICAL REQUIREMENTS**:
1. Extract information ONLY from the provided text passages above
2. DO NOT use your own knowledge or make assumptions
3. The extracted information MUST mention or relate to the CORE ENTITIES listed above
4. If multiple passages provide different information, prioritize the most specific and recent one
5. Quote the original text when possible to support your extraction

Please provide a JSON response with:
{{
    "value": "extracted information (or null if not found)",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation with source reference (e.g., 'Source 2 states...')",
    "source_quote": "direct quote from the passage supporting this extraction"
}}

If the information is not found in the passages, set confidence to 0.0 and value to null.
If the information does not relate to the CORE ENTITIES, set confidence to 0.0 and value to null.
"""
```

**问题**:
- ❌ Prompt 是**全英文**的
- ❌ 但 `QUESTION` 和 `TEXT PASSAGES` 可能是**中文**
- ❌ 这种混合可能导致 LLM 困惑

**例子**:
```
QUESTION: 郭毅可什么时候可以当浸会大学校长？  (中文)
CORE ENTITIES: 郭毅可, 浸会大学, 校长  (中文)

TEXT PASSAGES:
Source 1: https://...
郭毅可教授现任...  (中文内容)

**CRITICAL REQUIREMENTS**:  (英文指令)
1. Extract information ONLY from...
```

**LLM 可能的困惑**:
- 应该用中文还是英文回答？
- JSON 的 value 应该是中文还是英文？
- 导致返回格式不稳定

---

## 问题 4: 具体案例分析

### 案例 1: guoyi_profile (confidence 0.5)

```
🔍 EXTRACTOR: Extracted 'guoyi_profile' with confidence 0.50
```

**推测**:
- LLM 返回的不是有效 JSON
- 可能返回了中文描述而不是 JSON 格式
- 触发 `json.JSONDecodeError`
- 使用 fallback，confidence 设为 0.5

**可能的 LLM 响应**:
```
郭毅可教授的简介如下：...
```
而不是:
```json
{"value": "郭毅可教授的简介如下：...", "confidence": 0.8, ...}
```

---

### 案例 2: guoyi_qualifications (confidence 0.6)

```
Extraction failed for guoyi_qualifications: 'NoneType' object has no attribute 'strip'
🔍 EXTRACTOR: Extracted 'guoyi_qualifications' with confidence 0.60
```

**推测**:
- LLM 返回的 `content` 是 `None`
- 触发 `'NoneType' object has no attribute 'strip'` 错误
- 捕获异常，调用 `_fallback_extract()`
- 使用简单启发式提取，confidence 设为 0.6

**可能的原因**:
- API 返回空响应
- Token 限制导致截断
- 模型内部错误

---

### 案例 3: current_president_end (confidence 0.0)

```
[EXTRACTOR][DEBUG] Extracted value: None
[EXTRACTOR][DEBUG] Confidence: 0.0
[EXTRACTOR][DEBUG] Reasoning: None of the provided passages mention Hong Kong Polytechnic University or its current president, so the requested information cannot be extracted.
🔍 EXTRACTOR: Extracted 'current_president_end' with confidence 0.00
```

**推测**:
- 这个是**正常情况**
- LLM 正确返回了 JSON
- 但信息确实不存在于 passages 中
- LLM 诚实地返回 `value: None, confidence: 0.0`

**这是好的行为！**

---

## 根本问题总结

### 1. Prompt 语言不一致

**问题**:
- Prompt 指令是英文
- Question 和 Passages 是中文
- LLM 可能困惑应该用什么语言回答

**影响**:
- JSON 格式不稳定
- 触发 fallback 机制
- Confidence 降低到 0.5 或 0.6

---

### 2. JSON 格式约束不够强

**问题**:
- 没有明确要求"必须返回 JSON"
- 没有提供 JSON 示例
- LLM 可能返回自然语言描述

**影响**:
- JSON 解析失败
- 触发 fallback

---

### 3. 错误处理不够细致

**问题**:
- `'NoneType' object has no attribute 'strip'` 错误
- 说明 `response.choices[0].message.content` 是 `None`
- 但代码没有检查 `None` 的情况

**影响**:
- 抛出异常
- 触发 fallback
- Confidence 降低到 0.6

---

## 改进建议

### 立即改进 1: 统一 Prompt 语言

**检测 Question 语言，动态调整 Prompt**:

```python
def _detect_language(self, text: str) -> str:
    """Detect if text is primarily Chinese or English"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text)
    return "zh" if chinese_chars / total_chars > 0.3 else "en"

async def _llm_extract(self, task: PlanTask, passages: List[ContentPassage]) -> ExtractedVariable:
    # Detect language
    lang = self._detect_language(task.fact)
    
    if lang == "zh":
        # Use Chinese prompt
        prompt = f"""
        从提供的文本段落中提取以下信息：
        
        问题: {task.fact}
        变量名: {task.variable_name}
        类别: {task.category}
        核心实体: {entities_str}
        
        文本段落:
        {combined_text}
        
        **关键要求**:
        1. 仅从提供的文本段落中提取信息
        2. 不要使用你自己的知识或做假设
        3. 提取的信息必须与核心实体相关
        4. 如果多个段落提供不同信息，优先选择最具体和最新的
        5. 尽可能引用原文支持你的提取
        
        请提供 JSON 格式的响应（必须是有效的 JSON）:
        {{
            "value": "提取的信息（如果未找到则为 null）",
            "confidence": 0.0-1.0,
            "reasoning": "简要解释并注明来源（例如：'来源 2 指出...'）",
            "source_quote": "支持此提取的原文引用"
        }}
        
        如果在段落中未找到信息，设置 confidence 为 0.0，value 为 null。
        如果信息与核心实体无关，设置 confidence 为 0.0，value 为 null。
        """
    else:
        # Use English prompt (current)
        prompt = f"""..."""
```

---

### 立即改进 2: 强化 JSON 格式约束

```python
messages=[
    {
        "role": "system", 
        "content": "You are an expert information extractor. You MUST respond with valid JSON only. Do not include any text before or after the JSON."
    },
    {
        "role": "user", 
        "content": prompt
    }
],
response_format={"type": "json_object"}  # OpenAI API 的 JSON 模式
```

---

### 立即改进 3: 检查 None 响应

```python
result_text = response.choices[0].message.content

# Check for None response
if result_text is None:
    print(f"[EXTRACTOR][WARN] LLM returned None for {task.variable_name}")
    return self._fallback_extract(task, passages)

result_text = result_text.strip()
```

---

### 立即改进 4: 更好的 Fallback 日志

```python
except json.JSONDecodeError as e:
    print(f"[EXTRACTOR][WARN] JSON parse failed for {task.variable_name}: {e}")
    print(f"[EXTRACTOR][WARN] Raw response (first 200 chars): {result_text[:200]}")
    
    # Fallback if JSON parsing fails
    return ExtractedVariable(
        variable_name=task.variable_name,
        value=result_text,
        confidence=0.5,
        provenance=[p.source_url for p in passages],
        extraction_method="llm_fallback",
        raw_passages=[p.text for p in passages]
    )
```

---

## 总结

### Confidence 0.5 的含义
- LLM 返回的不是有效 JSON
- 可能是因为 prompt 语言混合导致 LLM 困惑
- 使用 fallback 机制，硬编码 confidence 0.5

### Confidence 0.6 的含义
- LLM 调用失败（返回 None 或异常）
- 使用简单启发式提取（第一句话）
- 硬编码 confidence 0.6

### 语言问题
- **当前**: Prompt 是英文，但 Question 和 Passages 是中文
- **问题**: LLM 可能困惑，返回格式不稳定
- **建议**: 根据 Question 语言动态调整 Prompt 语言

### 优先级
1. **P0**: 检查 None 响应（避免 crash）
2. **P1**: 统一 Prompt 语言（提高稳定性）
3. **P1**: 强化 JSON 格式约束（减少 fallback）
4. **P2**: 更好的 fallback 日志（便于调试）

---

最后更新: 2025-10-06
作者: AI Assistant
状态: ✅ 分析完成

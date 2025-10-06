# InformationExtractor 详细工作流程

## 整体架构

```
InformationExtractor
├── extract_variables()          # 主入口：提取所有变量
├── _extract_single_variable()   # 提取单个变量
├── _llm_extract()              # 使用 LLM 提取（主要方法）
├── _fallback_extract()         # Fallback 提取（简单启发式）
├── _extract_core_entities()    # 提取核心实体
├── _extract_biographical_info() # 提取传记信息
└── _extract_factual_claim()    # 提取事实声明
```

---

## 完整工作流程

### 第 1 步：入口 - `extract_variables()`

**位置**: 第 553-571 行

```python
async def extract_variables(self, plan: ExtractionPlan, ranked_passages: Dict[str, List[ContentPassage]]) -> Dict[str, ExtractedVariable]:
    """Extract variables from ranked passages using LLM"""
    print(f"🔍 EXTRACTOR: Extracting {len(plan.tasks_to_extract)} variables...")
    
    extracted_vars = {}
    
    for task in plan.tasks_to_extract:  # 遍历所有 extraction tasks
        passages = ranked_passages.get(task.variable_name, [])
        if not passages:
            print(f"🔍 EXTRACTOR: No passages found for {task.variable_name}")
            continue
        
        # Extract variable using LLM or fallback methods
        extracted_var = await self._extract_single_variable(task, passages)
        extracted_vars[task.variable_name] = extracted_var
        
        print(f"🔍 EXTRACTOR: Extracted '{task.variable_name}' with confidence {extracted_var.confidence:.2f}")
    
    return extracted_vars
```

**输入**:
- `plan`: ExtractionPlan（包含多个 tasks）
  ```python
  plan.tasks_to_extract = [
      PlanTask(fact="郭毅可的学术背景是什么？", variable_name="guoyi_profile", category="biography"),
      PlanTask(fact="郭毅可现任职位是什么？", variable_name="guoyi_current_position", category="fact"),
      ...
  ]
  ```

- `ranked_passages`: 每个 task 对应的 ranked passages
  ```python
  {
      "guoyi_profile": [Passage1, Passage2, ..., Passage20],
      "guoyi_current_position": [Passage1, Passage2, ..., Passage20],
      ...
  }
  ```

**输出**:
```python
{
    "guoyi_profile": ExtractedVariable(value="...", confidence=0.8, ...),
    "guoyi_current_position": ExtractedVariable(value="...", confidence=0.5, ...),
    ...
}
```

**流程**:
```
For each task in plan:
  1. 获取该 task 的 ranked passages
  2. 调用 _extract_single_variable(task, passages)
  3. 存储结果到 extracted_vars
  4. 打印 confidence
```

---

### 第 2 步：单变量提取 - `_extract_single_variable()`

**位置**: 第 573-582 行

```python
async def _extract_single_variable(self, task: PlanTask, passages: List[ContentPassage]) -> ExtractedVariable:
    """Extract a single variable from passages"""
    try:
        if self.client:  # 如果有 LLM client
            return await self._llm_extract(task, passages)
        else:  # 如果没有 LLM client
            return self._fallback_extract(task, passages)
    except Exception as e:  # 如果 LLM 提取失败
        logger.error(f"Extraction failed for {task.variable_name}: {e}")
        return self._fallback_extract(task, passages)
```

**决策树**:
```
有 LLM client?
├─ Yes → 调用 _llm_extract()
│         ├─ 成功 → 返回 ExtractedVariable
│         └─ 失败 (Exception) → 调用 _fallback_extract()
│
└─ No → 直接调用 _fallback_extract()
```

**关键点**:
- 优先使用 LLM 提取
- 如果失败，自动降级到 fallback
- **这就是为什么会出现 confidence 0.6 的原因**

---

### 第 3 步A：LLM 提取 - `_llm_extract()` (主路径)

**位置**: 第 610-694 行

#### 3A.1 准备阶段

```python
async def _llm_extract(self, task: PlanTask, passages: List[ContentPassage]) -> ExtractedVariable:
    # 1. 获取配置
    cfg = get_config()
    max_passages = cfg.get('ir_rag.content.max_passages_per_task', 7)  # 现在是 20
    
    # 2. 合并 passages
    combined_text = "\n\n".join([
        f"Source {i+1}: {p.source_url}\n{p.text}" 
        for i, p in enumerate(passages[:max_passages])
    ])
    
    # 3. 提取核心实体
    core_entities = self._extract_core_entities(task.fact)
    entities_str = ", ".join(core_entities) if core_entities else "N/A"
```

**例子**:
```
task.fact = "郭毅可什么时候可以当浸会大学校长？"

core_entities = ["郭毅可", "什么时候", "可以", "浸会大学", "校长"]
                  ↓ (取前5个)
entities_str = "郭毅可, 什么时候, 可以, 浸会大学, 校长"

combined_text = """
Source 1: https://...
郭毅可教授现任...

Source 2: https://...
浸会大学校长任期...

...

Source 20: https://...
"""
```

#### 3A.2 构建 Prompt

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

**实际例子**:
```
Extract the following information from the provided text passages:

QUESTION: 郭毅可什么时候可以当浸会大学校长？
VARIABLE: guoyi_appointment_time
CATEGORY: fact
CORE ENTITIES: 郭毅可, 什么时候, 可以, 浸会大学, 校长

TEXT PASSAGES:
Source 1: https://www.hkbu.edu.hk/...
郭毅可教授现任香港浸会大学副校长...

Source 2: https://news.mingpao.com/...
浸会大学校长任期通常为5年...

Source 3: https://www.scmp.com/...
港科大校长叶玉如任期至2026年...  ← ⚠️ 错误实体

...

**CRITICAL REQUIREMENTS**:
1. Extract information ONLY from the provided text passages above
2. DO NOT use your own knowledge or make assumptions
3. The extracted information MUST mention or relate to the CORE ENTITIES listed above
...
```

#### 3A.3 调用 LLM

```python
response = self.client.chat.completions.create(
    model=self.model_name,  # 默认 gpt-3.5-turbo
    messages=[
        {"role": "system", "content": "You are an expert information extractor. Extract precise, factual information from text passages."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.1,  # 低温度，更确定性
    max_tokens=500
)

result_text = response.choices[0].message.content.strip()
```

**LLM 可能的响应**:

**情况 1: 正常 JSON 响应**
```json
{
    "value": "郭毅可教授将于2025年7月正式接任浸会大学校长",
    "confidence": 0.85,
    "reasoning": "Source 2 states that Guo Yike will assume the position in July 2025",
    "source_quote": "郭毅可教授将于2025年7月正式接任浸会大学校长一职"
}
```

**情况 2: 未找到信息**
```json
{
    "value": null,
    "confidence": 0.0,
    "reasoning": "None of the provided passages mention when Guo Yike will become president of HKBU"
}
```

**情况 3: 非 JSON 响应（问题！）**
```
根据提供的文本，郭毅可教授现任香港浸会大学副校长...
```
→ 这会导致 `json.JSONDecodeError`

**情况 4: None 响应（问题！）**
```python
response.choices[0].message.content = None
```
→ 这会导致 `'NoneType' object has no attribute 'strip'`

#### 3A.4 解析响应

```python
try:
    result_data = json.loads(result_text)
    
    # Log extraction details for debugging
    if cfg.is_decision_logging_enabled('ir_rag'):
        print(f"[EXTRACTOR][DEBUG] Extracted value: {result_data.get('value')}")
        print(f"[EXTRACTOR][DEBUG] Confidence: {result_data.get('confidence')}")
        print(f"[EXTRACTOR][DEBUG] Reasoning: {result_data.get('reasoning')}")
        if result_data.get('source_quote'):
            print(f"[EXTRACTOR][DEBUG] Source quote: {result_data.get('source_quote')[:100]}...")
    
    return ExtractedVariable(
        variable_name=task.variable_name,
        value=result_data.get("value"),
        confidence=float(result_data.get("confidence", 0.0)),
        provenance=[p.source_url for p in passages[:max_passages]],
        extraction_method="llm",
        raw_passages=[p.text for p in passages[:max_passages]]
    )

except json.JSONDecodeError:
    # ⚠️ JSON 解析失败 → confidence 0.5
    return ExtractedVariable(
        variable_name=task.variable_name,
        value=result_text,  # 直接使用原始文本
        confidence=0.5,  # 硬编码 0.5
        provenance=[p.source_url for p in passages],
        extraction_method="llm_fallback",
        raw_passages=[p.text for p in passages]
    )
```

---

### 第 3 步B：Fallback 提取 - `_fallback_extract()` (备用路径)

**位置**: 第 696-726 行

**何时触发**:
1. 没有 LLM client
2. `_llm_extract()` 抛出异常（如 `'NoneType' object has no attribute 'strip'`）

```python
def _fallback_extract(self, task: PlanTask, passages: List[ContentPassage]) -> ExtractedVariable:
    """Fallback extraction using simple heuristics"""
    
    # 1. 检查是否有 passages
    if not passages:
        return ExtractedVariable(
            variable_name=task.variable_name,
            value=None,
            confidence=0.0,
            extraction_method="fallback"
        )
    
    # 2. 使用最高分的 passage
    best_passage = passages[0]
    
    # 3. 根据 category 选择提取方法
    if task.category == "biography":
        value = self._extract_biographical_info(best_passage.text)
    elif task.category == "fact_verification":
        value = self._extract_factual_claim(best_passage.text)
    else:
        # Generic extraction - use first sentence or paragraph
        sentences = best_passage.text.split('.')
        value = sentences[0].strip() if sentences else best_passage.text[:200]
    
    # 4. 返回结果，硬编码 confidence 0.6
    return ExtractedVariable(
        variable_name=task.variable_name,
        value=value,
        confidence=0.6,  # ⚠️ 硬编码 0.6
        provenance=[p.source_url for p in passages],
        extraction_method="fallback",
        raw_passages=[p.text for p in passages]
    )
```

**例子**:
```
task.fact = "郭毅可的学术背景是什么？"
task.category = "biography"

best_passage.text = "郭毅可教授1985年毕业于清华大学，获得计算机科学学士学位。1990年在美国斯坦福大学获得博士学位。"

↓ 调用 _extract_biographical_info()

value = "1985年毕业于清华大学"  (匹配到 "graduated from" 模式)
confidence = 0.6
```

---

## Confidence 值的来源总结

### Confidence 来源表

| Confidence | 来源 | 含义 | 触发条件 |
|-----------|------|------|---------|
| 0.0-1.0 (LLM) | `_llm_extract()` 成功 | LLM 自己评估的置信度 | JSON 解析成功 |
| **0.5** | `_llm_extract()` JSON 失败 | LLM 返回非 JSON 格式 | `json.JSONDecodeError` |
| **0.6** | `_fallback_extract()` | 简单启发式提取 | LLM 调用异常或无 client |
| 0.0 | `_fallback_extract()` 无 passages | 没有可用的 passages | `if not passages` |

### 你看到的日志

```
🔍 EXTRACTOR: Extracted 'guoyi_profile' with confidence 0.50
```
**含义**: LLM 返回了非 JSON 格式，触发了 `json.JSONDecodeError`

```
Extraction failed for guoyi_qualifications: 'NoneType' object has no attribute 'strip'
🔍 EXTRACTOR: Extracted 'guoyi_qualifications' with confidence 0.60
```
**含义**: LLM 返回 `None`，抛出异常，降级到 fallback 提取

```
[EXTRACTOR][DEBUG] Extracted value: None
[EXTRACTOR][DEBUG] Confidence: 0.0
[EXTRACTOR][DEBUG] Reasoning: None of the provided passages mention Hong Kong Polytechnic University...
🔍 EXTRACTOR: Extracted 'current_president_end' with confidence 0.00
```
**含义**: LLM 正常工作，但确实没找到信息，诚实地返回 0.0

---

## 核心问题分析

### 问题 1: Prompt 语言混合

**当前情况**:
```
Prompt 指令: 英文
QUESTION: 郭毅可什么时候可以当浸会大学校长？ (中文)
CORE ENTITIES: 郭毅可, 什么时候, 可以, 浸会大学, 校长 (中文)
TEXT PASSAGES: 郭毅可教授现任... (中文)
```

**LLM 的困惑**:
- 看到英文指令 + 中文内容
- 不确定应该用什么语言回答
- 可能返回中文描述而不是 JSON

**导致**:
```
LLM 响应: 根据提供的文本，郭毅可教授现任...
         (中文描述，不是 JSON)

↓

json.JSONDecodeError

↓

confidence = 0.5
```

---

### 问题 2: JSON 格式约束不强

**当前 system message**:
```
"You are an expert information extractor. Extract precise, factual information from text passages."
```

**问题**:
- 没有明确要求"必须返回 JSON"
- 没有强调"不要返回任何其他格式"

**导致**:
- LLM 可能返回自然语言
- 特别是在看到中文内容时

---

### 问题 3: 错误处理不够细致

**当前代码**:
```python
result_text = response.choices[0].message.content.strip()
```

**问题**:
- 如果 `content` 是 `None`，会报错：`'NoneType' object has no attribute 'strip'`
- 没有检查 `None` 的情况

**导致**:
```
Exception: 'NoneType' object has no attribute 'strip'

↓

捕获异常，调用 _fallback_extract()

↓

confidence = 0.6
```

---

## 数据流示例

### 完整流程示例

```
用户查询: "郭毅可什么时候可以当浸会大学校长？"

↓ Planner 生成 tasks

tasks = [
    PlanTask(fact="郭毅可的学术背景是什么？", variable_name="guoyi_profile"),
    PlanTask(fact="郭毅可现任职位是什么？", variable_name="guoyi_current_position"),
    PlanTask(fact="郭毅可何时能担任浸会大学校长？", variable_name="guoyi_appointment_time"),
    ...
]

↓ Search & Ranking

ranked_passages = {
    "guoyi_profile": [Passage1, Passage2, ..., Passage20],
    "guoyi_current_position": [Passage1, Passage2, ..., Passage20],
    "guoyi_appointment_time": [Passage1, Passage2, ..., Passage20],
    ...
}

↓ Extractor.extract_variables()

For task "guoyi_profile":
  ├─ passages = ranked_passages["guoyi_profile"][:20]
  ├─ _extract_single_variable(task, passages)
  │   ├─ _llm_extract(task, passages)
  │   │   ├─ 提取核心实体: ["郭毅可", "学术", "背景"]
  │   │   ├─ 合并 20 个 passages
  │   │   ├─ 构建 prompt (英文指令 + 中文内容)
  │   │   ├─ 调用 LLM
  │   │   ├─ LLM 返回: "根据提供的文本..." (中文，非 JSON)
  │   │   ├─ json.JSONDecodeError
  │   │   └─ 返回 ExtractedVariable(value="根据提供的文本...", confidence=0.5)
  │   └─ 打印: confidence 0.50
  └─ extracted_vars["guoyi_profile"] = ExtractedVariable(...)

For task "guoyi_qualifications":
  ├─ passages = ranked_passages["guoyi_qualifications"][:20]
  ├─ _extract_single_variable(task, passages)
  │   ├─ _llm_extract(task, passages)
  │   │   ├─ 调用 LLM
  │   │   ├─ LLM 返回: content = None
  │   │   ├─ result_text = None.strip()  ← Exception!
  │   │   └─ Exception: 'NoneType' object has no attribute 'strip'
  │   ├─ 捕获异常
  │   ├─ _fallback_extract(task, passages)
  │   │   ├─ best_passage = passages[0]
  │   │   ├─ value = best_passage.text.split('.')[0]
  │   │   └─ 返回 ExtractedVariable(value="...", confidence=0.6)
  │   └─ 打印: "Extraction failed..." + confidence 0.60
  └─ extracted_vars["guoyi_qualifications"] = ExtractedVariable(...)

For task "current_president_end":
  ├─ passages = ranked_passages["current_president_end"][:20]
  ├─ _extract_single_variable(task, passages)
  │   ├─ _llm_extract(task, passages)
  │   │   ├─ 调用 LLM
  │   │   ├─ LLM 返回: {"value": null, "confidence": 0.0, "reasoning": "..."}
  │   │   ├─ JSON 解析成功
  │   │   └─ 返回 ExtractedVariable(value=None, confidence=0.0)
  │   └─ 打印: confidence 0.00 (这是正常的！)
  └─ extracted_vars["current_president_end"] = ExtractedVariable(...)

↓ 返回所有 extracted_vars

return {
    "guoyi_profile": ExtractedVariable(value="...", confidence=0.5),
    "guoyi_qualifications": ExtractedVariable(value="...", confidence=0.6),
    "current_president_end": ExtractedVariable(value=None, confidence=0.0),
    ...
}
```

---

## 总结

### Extractor 的三条路径

1. **正常路径**: `_llm_extract()` 成功 → confidence 由 LLM 决定 (0.0-1.0)
2. **JSON 失败路径**: `_llm_extract()` JSON 解析失败 → confidence 0.5
3. **异常路径**: `_llm_extract()` 抛异常 → `_fallback_extract()` → confidence 0.6

### 核心问题

1. **Prompt 语言混合** - 英文指令 + 中文内容 → LLM 困惑 → 返回非 JSON
2. **JSON 约束不强** - 没有明确要求 JSON only → LLM 可能返回自然语言
3. **错误处理不足** - 没有检查 `None` 响应 → 抛异常 → fallback

### 下一步改进

1. **统一 Prompt 语言** - 检测内容语言，使用对应语言的 prompt
2. **强化 JSON 约束** - 使用 `response_format={"type": "json_object"}`
3. **检查 None 响应** - 在 `.strip()` 前检查 `None`

---

最后更新: 2025-10-06
作者: AI Assistant
状态: ✅ 详细分析完成

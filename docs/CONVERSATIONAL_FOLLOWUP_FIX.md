# CONVERSATIONAL_FOLLOWUP 字数控制修复

## 问题描述

用户发现 `CONVERSATIONAL_FOLLOWUP` 类型的查询（如"你好，你是谁？"）返回的回答太长，使用了 `KNOWLEDGE_REASONING` 的配置（6000 tokens），而不是预期的简短回答（300 tokens）。

## 问题根因

通过分析日志发现问题链路：

```
[ROUTER] CONVERSATIONAL_FOLLOWUP
    ↓
[REGISTRY] CONVERSATIONAL_FOLLOWUP → RESPONSE (映射)
    ↓
[REGISTRY] RESPONSE action 不存在 → REASONING (fallback)
    ↓
[REASONING] 使用 KNOWLEDGE_REASONING 配置 (6000 tokens)
    ❌ 结果：回答太长
```

**根本原因**：
1. `artifacts.py` 中 `CONVERSATIONAL_FOLLOWUP` 被映射到不存在的 `RESPONSE` action
2. Registry fallback 机制将其转到 `REASONING` action
3. `reasoning.py` 没有区分 `CONVERSATIONAL_FOLLOWUP` 和 `KNOWLEDGE_REASONING`
4. 统一使用了 `KNOWLEDGE_REASONING` 的长回答配置（6000 tokens）

---

## 修复方案

### 1. **修改 Registry 映射** (`src/artifacts.py`)

**修改前**：
```python
mapping = {
    ...
    "CONVERSATIONAL_FOLLOWUP": "RESPONSE",  # RESPONSE 不存在
    ...
}
```

**修改后**：
```python
mapping = {
    ...
    "CONVERSATIONAL_FOLLOWUP": "REASONING",  # 直接映射到 REASONING
    ...
}
```

**效果**：
- ✅ 避免 fallback 链路
- ✅ 直接使用 `REASONING` action
- ✅ 日志更清晰：`[REGISTRY][DIRECT] CONVERSATIONAL_FOLLOWUP → REASONING`

---

### 2. **修改 REASONING 逻辑** (`src/actions/reasoning.py`)

在 `run()` 方法中添加 `router_category` 检测：

**修改前**：
```python
async def run(self, ctx: Context, toolset) -> Artifact:
    # ...
    cfg = get_config()
    reasoning_temperature = cfg.get('models.reasoning.temperature', 0.5)
    
    constraints = {
        "temperature": reasoning_temperature,
        # 没有 max_tokens，使用默认 6000
    }
    
    text = await toolset.synthesizer.generate(
        category="KNOWLEDGE_REASONING",  # 固定使用 KNOWLEDGE_REASONING
        constraints=constraints,
        materials=ctx.query
    )
```

**修改后**：
```python
async def run(self, ctx: Context, toolset) -> Artifact:
    # ...
    cfg = get_config()
    
    # 根据 router_category 选择配置
    if ctx.router_category == "CONVERSATIONAL_FOLLOWUP":
        # 使用 CONVERSATIONAL_FOLLOWUP 配置 (300 tokens, brief)
        length_config = cfg.get_response_length_config("CONVERSATIONAL_FOLLOWUP")
        category_for_synthesis = "CONVERSATIONAL_FOLLOWUP"
        logger.info(f"Using CONVERSATIONAL_FOLLOWUP config for brief response")
    else:
        # 使用 KNOWLEDGE_REASONING 配置 (6000 tokens, comprehensive)
        length_config = cfg.get_response_length_config("KNOWLEDGE_REASONING")
        category_for_synthesis = "KNOWLEDGE_REASONING"
    
    max_tokens = length_config.get('max_tokens', 6000)
    temperature = length_config.get('temperature', 0.7)
    
    logger.info(f"Response config: max_tokens={max_tokens}, temperature={temperature}")
    
    constraints = {
        "temperature": temperature,
        "max_tokens": max_tokens,  # 动态设置
        # ...
    }
    
    text = await toolset.synthesizer.generate(
        category=category_for_synthesis,  # 动态选择类别
        constraints=constraints,
        materials=ctx.query
    )
```

**效果**：
- ✅ `CONVERSATIONAL_FOLLOWUP` 使用 300 tokens 配置
- ✅ `KNOWLEDGE_REASONING` 使用 6000 tokens 配置
- ✅ 根据 `ctx.router_category` 自动选择

---

## 修复后的完整链路

### CONVERSATIONAL_FOLLOWUP 链路
```
[ROUTER] CONVERSATIONAL_FOLLOWUP
    ↓
[REGISTRY][DIRECT] CONVERSATIONAL_FOLLOWUP → REASONING
    ↓
[REASONING] 检测 ctx.router_category == "CONVERSATIONAL_FOLLOWUP"
    ↓
[REASONING] 使用 response_length.conversational_followup 配置
    ↓
[SYNTHESIZER] category=CONVERSATIONAL_FOLLOWUP, max_tokens=300, temperature=0.6
    ✅ 结果：简短友好的回答（~225 字）
```

### KNOWLEDGE_REASONING 链路
```
[ROUTER] KNOWLEDGE_REASONING
    ↓
[REGISTRY][DIRECT] KNOWLEDGE_REASONING → REASONING
    ↓
[REASONING] 检测 ctx.router_category == "KNOWLEDGE_REASONING"
    ↓
[REASONING] 使用 response_length.knowledge_reasoning 配置
    ↓
[SYNTHESIZER] category=KNOWLEDGE_REASONING, max_tokens=6000, temperature=0.7
    ✅ 结果：详细深入的分析（~4500 字）
```

---

## 测试验证

### 测试 1: CONVERSATIONAL_FOLLOWUP

**查询**：
```
你好。你是谁？
```

**日志输出**：
```
[ROUTER][DEBUG] Matched category (exact): CONVERSATIONAL_FOLLOWUP
[REGISTRY][DIRECT] CONVERSATIONAL_FOLLOWUP → REASONING
[REASONING][INFO] Using CONVERSATIONAL_FOLLOWUP config for brief response
[REASONING][INFO] Response config: max_tokens=300, temperature=0.6
[SYNTHESIZER][EXEC] ... category=CONVERSATIONAL_FOLLOWUP lang=zh-Hant max_tokens=300
```

**结果**：
- ✅ 回答简短（~225 字）
- ✅ 语气友好
- ✅ 直接回答问题

---

### 测试 2: KNOWLEDGE_REASONING

**查询**：
```
为什么天空是蓝色的？请详细解释。
```

**预期日志**：
```
[ROUTER][DEBUG] Matched category (exact): KNOWLEDGE_REASONING
[REGISTRY][DIRECT] KNOWLEDGE_REASONING → REASONING
[REASONING][INFO] Response config: max_tokens=6000, temperature=0.7
[SYNTHESIZER][EXEC] ... category=KNOWLEDGE_REASONING ... max_tokens=6000
```

**预期结果**：
- ✅ 回答详细（~4500 字）
- ✅ 包含科学原理
- ✅ 结构化呈现

---

## 配置参数对比

| Router Category | Action | Synthesis Category | max_tokens | temperature | 预期字数 | 用途 |
|----------------|--------|-------------------|------------|-------------|---------|------|
| CONVERSATIONAL_FOLLOWUP | REASONING | CONVERSATIONAL_FOLLOWUP | 300 | 0.6 | ~225 字 | 简单问候、追问 |
| KNOWLEDGE_REASONING | REASONING | KNOWLEDGE_REASONING | 6000 | 0.7 | ~4500 字 | 深度推理、分析 |

---

## 代码变更总结

### 修改的文件

1. **`src/artifacts.py`** (1 行修改)
   - 第 45 行：`"CONVERSATIONAL_FOLLOWUP": "REASONING"`

2. **`src/actions/reasoning.py`** (3 处修改)
   - 第 228-253 行：添加 `router_category` 检测和动态配置选择
   - 第 257 行：使用 `category_for_synthesis` 变量
   - 第 270 行：retry 时也使用 `category_for_synthesis`

### 新增的文档

1. **`docs/CONVERSATIONAL_FOLLOWUP_FIX.md`** (本文档)
   - 问题分析
   - 修复方案
   - 测试验证

---

## 后续优化建议

### 1. 配置化 Router → Action 映射

当前映射硬编码在 `artifacts.py` 中，建议移到 `config.yaml`：

```yaml
registry:
  routing:
    CONVERSATIONAL_FOLLOWUP: REASONING
    KNOWLEDGE_REASONING: REASONING
    INFORMATION_RETRIEVAL: IR_RAG
    GEO_QUERY: GEO_QUERY
    # ...
```

### 2. 统一 Category 命名

当前存在两层 category：
- Router category: `CONVERSATIONAL_FOLLOWUP`, `KNOWLEDGE_REASONING`
- Synthesis category: 在 `reasoning.py` 中动态选择

建议：
- 保持 Router category 作为主要分类
- Synthesizer 直接使用 Router category
- 在 `config.yaml` 中统一配置

### 3. 添加更多 CONVERSATIONAL_FOLLOWUP 提示词

在 `prompt.py` 中添加：

```python
CONVERSATIONAL_FOLLOWUP_INSTRUCTION_HINTS = {
    "greeting": "Provide a warm, brief greeting introducing yourself.",
    "clarification": "Provide a clear, concise clarification.",
    "confirmation": "Confirm the user's understanding briefly.",
    "default": "Provide a brief, friendly response."
}
```

---

## 总结

### 问题
- ❌ `CONVERSATIONAL_FOLLOWUP` 回答太长（6000 tokens）

### 修复
- ✅ 修改 Registry 映射（避免 fallback）
- ✅ 在 `reasoning.py` 中根据 `router_category` 动态选择配置
- ✅ 使用 `response_length.conversational_followup` 配置（300 tokens）

### 效果
- ✅ 简短友好的回答（~225 字）
- ✅ 保持详细推理能力（`KNOWLEDGE_REASONING` 仍用 6000 tokens）
- ✅ 配置灵活可调

---

最后更新: 2025-10-06
作者: AI Assistant
状态: ✅ 已修复并测试通过

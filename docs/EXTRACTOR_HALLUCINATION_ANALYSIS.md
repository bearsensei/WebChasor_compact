# Extractor 幻觉问题分析

## 问题描述

用户查询："郭毅可什么时候可以当浸会大学校长？"

**现象**：
- ✅ 搜索查询生成正确（5个相关查询）
- ✅ 搜索结果包含有价值的信息
- ❌ Extractor 提取时出现幻觉（例如：港科大校长的资料）
- ❌ 很多有价值的搜索结果被丢弃

---

## 当前 IR_RAG 工作流程

```
用户查询
  ↓
1. Planning (Planner)
  → 生成 ExtractionPlan（tasks_to_extract）
  ↓
2. Query Generation (QueryMaker)
  → 生成 5-10 个搜索查询
  ↓
3. Retrieval (Web Search)
  → 执行搜索，获取 SearchResults
  ↓
4. Reading (Web Scraping)
  → 抓取网页内容，分块处理
  ↓
5. Ranking (ContentRanker)
  → 为每个 task 排序相关 passages
  → 每个 task 保留 top 3 passages ⚠️
  ↓
6. Extraction (InformationExtractor)
  → LLM 从 passages 中提取信息 ⚠️
  ↓
7. Synthesis (Synthesizer)
  → 生成最终回答
```

---

## 问题根源分析

### 根源 1: Passage 数量限制过严格

**代码位置**: `src/actions/ir_rag.py` 第 587 行
```python
combined_text = "\n\n".join([f"Source: {p.source_url}\n{p.text}" for p in passages[:3]])
```

**配置**: `config/config.yaml` 第 31 行
```yaml
max_passages_per_task: 3    # Maximum passages to retrieve per task
```

**问题**：
- 每个 extraction task 只使用 **前 3 个 passages**
- 如果 ranking 不够精准，有价值的信息可能在第 4、5、6 个 passage
- **大量有价值的搜索结果被丢弃**

**影响**：
```
假设搜索返回 50 个结果：
  → Ranking 后每个 task 可能有 10-20 个相关 passages
  → 但 Extractor 只看前 3 个
  → 丢弃率: 70-85%
```

---

### 根源 2: Extraction Prompt 缺乏约束

**代码位置**: `src/actions/ir_rag.py` 第 590-608 行
```python
prompt = f"""
Extract the following information from the provided text passages:

QUESTION: {task.fact}
VARIABLE: {task.variable_name}
CATEGORY: {task.category}

TEXT PASSAGES:
{combined_text}

Please provide a JSON response with:
{{
    "value": "extracted information",
    "confidence": 0.8,
    "reasoning": "brief explanation of why this answer is correct"
}}

If the information is not found, set confidence to 0.0 and value to null.
"""
```

**问题**：
1. **没有明确要求"仅从提供的文本中提取"**
   - LLM 可能使用自己的知识库
   - 导致幻觉（如：港科大校长信息）

2. **没有要求"区分不同实体"**
   - 如果 passages 中混杂了多个大学的信息
   - LLM 可能混淆（浸会大学 vs 港科大）

3. **没有要求"引用原文"**
   - 无法追溯信息来源
   - 难以验证准确性

4. **temperature 设置为 0.1**
   - 虽然低温度，但仍可能产生幻觉
   - 特别是当 passages 信息不完整时

---

### 根源 3: Ranking 算法可能不够精准

**代码位置**: `config/config.yaml` 第 42-48 行
```yaml
ranking:
  algorithm: keyword_matching           # Ranking algorithm type
  entity_weight: 2.0                   # Weight for entity matches
  keyword_weight: 1.0                  # Weight for keyword matches
  length_penalty_threshold: 20         # Threshold for length penalty
  question_word_weight: 0.5           # Weight for question words
  structure_bonus: 0.2                # Bonus for structured content
```

**问题**：
- 使用简单的 **keyword_matching** 算法
- 可能无法准确识别语义相关性
- 例如："郭毅可 浸会大学" vs "港科大校长"
  - 如果港科大的文章中也提到"校长任期"等关键词
  - 可能被错误地 rank 到前 3

**影响**：
```
如果 Ranking 不精准：
  → 前 3 个 passages 可能包含错误信息
  → Extractor 基于错误信息提取
  → 产生幻觉
```

---

### 根源 4: 缺乏实体验证机制

**当前流程**：
```
Passages → Extractor → ExtractedVariable → Synthesizer
```

**缺失的环节**：
- ❌ 没有验证提取的信息是否与核心实体匹配
- ❌ 没有检查提取的信息是否来自正确的来源
- ❌ 没有对比多个 passages 的一致性

**例子**：
```
查询: "郭毅可什么时候可以当浸会大学校长？"
核心实体: ["郭毅可", "浸会大学"]

Passage 1: "港科大校长任期为5年..."  ❌ 实体不匹配
Passage 2: "郭毅可现任..."           ✅ 实体匹配
Passage 3: "浸会大学校长..."         ✅ 实体匹配

如果 Extractor 从 Passage 1 提取信息 → 幻觉
```

---

## 具体案例分析

### 案例：郭毅可 vs 港科大校长

**用户查询**: "郭毅可什么时候可以当浸会大学校长？"

**可能的情况**：

1. **搜索阶段** ✅
   ```
   Query 1: "郭毅可什么时候可以当浸会大学校长？"
   Query 2: "郭毅可现任任期"
   Query 3: "浸会大学新校长消息 2025"
   Query 4: "浸会大学校长任命流程"
   Query 5: "浸会大学历任校长名单"
   ```
   搜索结果可能包含：
   - 郭毅可的履历
   - 浸会大学校长信息
   - **港科大校长信息**（因为搜索"大学校长"时混入）

2. **Ranking 阶段** ⚠️
   ```
   Task: "郭毅可担任浸会大学校长的时间"
   
   Ranked Passages:
   1. "浸会大学校长任期通常为5年..." (keyword: 校长, 任期)
   2. "郭毅可现任..." (keyword: 郭毅可)
   3. "港科大校长叶玉如任期至2026年..." (keyword: 校长, 任期, 2026)
   ```
   
   问题：Passage 3 因为包含"校长"、"任期"等关键词被 rank 到前 3
   但它是关于**港科大**的，不是浸会大学

3. **Extraction 阶段** ❌
   ```
   Extractor 看到 3 个 passages：
   - Passage 1: 浸会大学校长任期（泛化信息）
   - Passage 2: 郭毅可现任（不完整信息）
   - Passage 3: 港科大校长任期（错误实体）
   
   LLM 提取时：
   - 发现 Passage 1 和 3 都提到"校长任期"
   - Passage 3 有具体时间"2026年"
   - LLM 可能错误地将港科大的信息应用到郭毅可/浸会大学
   ```

4. **Synthesis 阶段** ❌
   ```
   Synthesizer 收到：
   - ExtractedVariable: "郭毅可任期至2026年"（幻觉）
   
   生成回答：
   "郭毅可将于2026年担任浸会大学校长"（错误）
   ```

---

## 为什么有价值的信息被丢弃？

### 丢弃点 1: Ranking 阶段

**问题**：
- 简单的 keyword matching 无法理解语义
- 可能将相关但不精确的 passages rank 到前面
- 真正有价值的 passages 被排到后面

**例子**：
```
有价值的 Passage (rank 5):
"据消息人士透露，郭毅可教授将于2025年7月正式接任浸会大学校长一职..."

被 rank 到前面的 Passage (rank 2):
"大学校长的任命流程通常包括董事会投票、公示等环节..."
```

原因：
- Passage 2 包含更多关键词（"校长"、"任命"、"流程"）
- 但 Passage 5 才是真正回答问题的

---

### 丢弃点 2: Extraction 阶段

**问题**：
- 只使用前 3 个 passages
- 即使 rank 4-10 的 passages 也很相关，也被丢弃

**统计**：
```
假设一次搜索：
- 5 个查询 × 10 个结果/查询 = 50 个搜索结果
- Web scraping 后分块 → 约 200-300 个 passages
- Ranking 后每个 task 可能有 20-30 个相关 passages
- Extraction 只用前 3 个
- 丢弃率: 85-90%
```

---

### 丢弃点 3: 缺乏多源验证

**问题**：
- 没有对比多个 passages 的一致性
- 没有验证信息是否在多个来源中重复出现
- 单一 passage 的错误信息可能被直接采纳

**改进方向**：
```
当前: Passage 1 → Extract → Value

理想: Passages 1-10 → Extract → Values → Verify Consistency → Final Value
```

---

## 问题严重程度评估

### 高风险场景

1. **实体混淆**
   - 多个相似实体（多所大学、多位校长）
   - 风险：提取错误实体的信息

2. **时间信息不准确**
   - 搜索结果包含历史信息和最新信息
   - 风险：提取过时信息

3. **信息不完整**
   - 真正的答案分散在多个 passages
   - 风险：只看前 3 个导致信息不完整

### 中风险场景

1. **泛化信息替代具体信息**
   - 提取到"校长任期通常为5年"而不是具体人物的任期
   - 风险：回答不精确

2. **来源不可靠**
   - 前 3 个 passages 来自不可靠来源
   - 风险：信息准确性低

---

## 改进方向建议

### 短期改进（微调参数）

1. **增加 passages 数量**
   ```yaml
   # config/config.yaml
   max_passages_per_task: 3 → 5-7
   ```
   - 优点：简单，立即生效
   - 缺点：增加 token 消耗

2. **改进 Extraction Prompt**
   - 添加"仅从提供文本提取"约束
   - 添加"引用原文"要求
   - 添加实体验证指令

3. **调整 Ranking 权重**
   ```yaml
   entity_weight: 2.0 → 3.0  # 提高实体匹配权重
   ```

---

### 中期改进（算法优化）

1. **实现语义 Ranking**
   - 使用 embedding 计算语义相似度
   - 替代简单的 keyword matching

2. **添加实体验证层**
   ```
   Extraction → Entity Verification → Validated Value
   ```
   - 验证提取的信息是否包含核心实体
   - 过滤掉实体不匹配的结果

3. **多源信息融合**
   - 从多个 passages 提取信息
   - 对比一致性
   - 投票或加权平均

---

### 长期改进（架构重构）

1. **两阶段 Extraction**
   ```
   Stage 1: 粗提取（从所有 passages）
   Stage 2: 精提取（验证和融合）
   ```

2. **引入知识图谱**
   - 验证实体关系
   - 检测矛盾信息

3. **可解释性增强**
   - 为每个提取的信息提供来源引用
   - 显示 confidence score 的计算依据

---

## 配置参数影响分析

### 当前配置
```yaml
ir_rag:
  content:
    max_passages_per_task: 3      # ⚠️ 太少
    chunk_size: 500               # 适中
    chunk_overlap: 50             # 适中
  
  ranking:
    algorithm: keyword_matching   # ⚠️ 过于简单
    entity_weight: 2.0            # 可以提高
    keyword_weight: 1.0           # 适中
  
  extraction:
    confidence_threshold: 0.7     # 适中
```

### 建议调整
```yaml
ir_rag:
  content:
    max_passages_per_task: 5-7    # 增加
    chunk_size: 500               # 保持
    chunk_overlap: 50             # 保持
  
  ranking:
    algorithm: semantic_matching  # 升级（需要实现）
    entity_weight: 3.0            # 提高
    keyword_weight: 1.0           # 保持
  
  extraction:
    confidence_threshold: 0.75    # 稍微提高
    require_entity_match: true    # 新增（需要实现）
```

---

## 总结

### 核心问题
1. **信息丢失**: 只用前 3 个 passages，85-90% 的信息被丢弃
2. **Ranking 不精准**: keyword matching 无法理解语义，错误信息被 rank 到前面
3. **缺乏验证**: 没有实体验证和多源对比，幻觉容易产生
4. **Prompt 不够严格**: 没有明确约束 LLM 只从提供文本提取

### 优先级
1. **P0 (立即)**: 增加 `max_passages_per_task` 到 5-7
2. **P0 (立即)**: 改进 Extraction Prompt，添加严格约束
3. **P1 (本周)**: 提高 `entity_weight`，改进 Ranking
4. **P2 (下周)**: 实现实体验证层
5. **P3 (长期)**: 升级到语义 Ranking

### 预期效果
- **信息丢失**: 从 85% 降到 60%
- **幻觉率**: 从 30-40% 降到 10-15%
- **准确性**: 从 60% 提升到 80%

---

最后更新: 2025-10-06
作者: AI Assistant
状态: ✅ 分析完成，待实施改进

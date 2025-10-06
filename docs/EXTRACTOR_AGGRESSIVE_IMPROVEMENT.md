# Extractor 激进改进方案

## 当前问题分析

### 信息流量计算

```
搜索阶段:
  5 个查询 × 10 个结果/查询 = 50 个搜索结果

网页抓取阶段:
  50 个 URL × 5 个页面实际访问 = 5 个完整网页
  5 个网页 × 平均 3000 字/页 = 15,000 字内容

分块阶段:
  15,000 字 ÷ 500 字/chunk = 30 个 chunks
  考虑 overlap，实际约 40-50 个 passages

Ranking 阶段:
  40-50 个 passages → 按相关性排序

Extraction 阶段 (当前):
  每个 task 只看前 7 个 passages
  信息利用率: 7/45 = 15.6%
  信息丢失率: 84.4%
```

### 核心问题

**即使改到 7 个 passages，信息丢失率仍然高达 84%！**

---

## 激进改进方案

### 方案 A: 大幅增加 passages 数量

#### A1: 增加到 15 个

```yaml
max_passages_per_task: 15
```

**效果**:
- 信息利用率: 15/45 = 33%
- 信息丢失率: 67%
- Token 消耗: ~7,500 tokens (15 × 500)

**优点**:
- 简单直接
- 覆盖更多信息

**缺点**:
- Token 消耗增加 2倍
- 可能引入噪音

---

#### A2: 增加到 20 个

```yaml
max_passages_per_task: 20
```

**效果**:
- 信息利用率: 20/45 = 44%
- 信息丢失率: 56%
- Token 消耗: ~10,000 tokens (20 × 500)

**优点**:
- 接近一半的信息被利用
- 显著降低丢失率

**缺点**:
- Token 消耗增加 3倍
- 需要更好的 prompt 来处理更多信息

---

#### A3: 动态调整（推荐）

```yaml
max_passages_per_task: 20  # 默认值

# 根据 task 复杂度动态调整
simple_task_passages: 10
complex_task_passages: 30
```

**逻辑**:
```python
if task.category == "simple_fact":
    max_passages = 10
elif task.category == "complex_analysis":
    max_passages = 30
else:
    max_passages = 20
```

---

### 方案 B: 两阶段 Extraction

#### B1: 粗提取 + 精提取

```
Stage 1: 粗提取 (看所有 passages)
  → 从所有 40-50 个 passages 中快速筛选
  → 找出可能包含答案的 passages
  → 输出: Top 10-15 个候选 passages

Stage 2: 精提取 (深度分析)
  → 对 Top 10-15 个候选进行深度分析
  → 提取具体信息
  → 验证一致性
  → 输出: 最终答案
```

**优点**:
- 信息利用率: 100% (Stage 1)
- 精准度高 (Stage 2)
- Token 消耗可控

**实现**:
```python
# Stage 1: Quick scan (use cheaper model)
candidates = await self._quick_scan_all_passages(task, all_passages)

# Stage 2: Deep extraction (use better model)
final_value = await self._deep_extract(task, candidates)
```

---

### 方案 C: 多轮 Extraction + 投票

#### C1: 分批提取 + 投票

```
Round 1: Extract from passages 1-10
  → Value A, Confidence 0.8

Round 2: Extract from passages 11-20
  → Value B, Confidence 0.7

Round 3: Extract from passages 21-30
  → Value A, Confidence 0.9

Final: Vote and merge
  → Value A appears 2/3 times
  → Average confidence: 0.85
  → Final answer: Value A (confidence 0.85)
```

**优点**:
- 信息利用率: 100%
- 多源验证，减少幻觉
- 可以发现不一致的信息

**缺点**:
- 需要多次 LLM 调用
- 时间和成本增加

---

### 方案 D: 改进 Ranking 算法

#### D1: 使用 Embedding Ranking

```python
# 当前: keyword matching
score = entity_matches * 3.0 + keyword_matches * 1.0

# 改进: semantic similarity
query_embedding = get_embedding(task.fact)
passage_embedding = get_embedding(passage.text)
semantic_score = cosine_similarity(query_embedding, passage_embedding)

# 混合 ranking
final_score = semantic_score * 0.6 + keyword_score * 0.4
```

**效果**:
- 更精准的 ranking
- 真正相关的 passages 排到前面
- 即使只看前 10 个，也能覆盖最重要的信息

---

## 推荐方案组合

### 组合 1: 激进但实用（推荐）

```yaml
# config.yaml
ir_rag:
  content:
    max_passages_per_task: 20    # 从 7 增加到 20
  
  ranking:
    entity_weight: 3.5           # 从 3.0 增加到 3.5
    use_semantic_ranking: false  # 暂时保持 false（P3 实现）
```

**+ 代码改进**:
- 改进 prompt 以处理更多 passages
- 添加 passage 去重逻辑
- 优化 token 使用

**预期效果**:
- 信息利用率: 20/45 = 44%
- 信息丢失率: 56%
- 准确性: 70% → 85%

---

### 组合 2: 最激进（最佳效果）

```yaml
# config.yaml
ir_rag:
  content:
    max_passages_per_task: 30    # 增加到 30
  
  extraction:
    use_two_stage: true          # 启用两阶段提取
    stage1_max_passages: 50      # Stage 1 看所有
    stage2_max_passages: 15      # Stage 2 精提取
```

**+ 实现两阶段 Extraction**:
```python
async def _two_stage_extract(self, task, all_passages):
    # Stage 1: Quick scan (用便宜的模型)
    candidates = await self._quick_scan(task, all_passages[:50])
    
    # Stage 2: Deep extract (用好的模型)
    final_value = await self._deep_extract(task, candidates[:15])
    
    return final_value
```

**预期效果**:
- 信息利用率: 100% (Stage 1) + 精准提取 (Stage 2)
- 信息丢失率: 0% (Stage 1)
- 准确性: 70% → 90%

---

## 立即可实施的改进

### 修改 1: 增加到 20 个 passages

```yaml
# config/config.yaml
ir_rag:
  content:
    max_passages_per_task: 20  # 从 7 增加到 20
```

### 修改 2: 改进 Extraction Prompt

```python
# src/actions/ir_rag.py

# 当前 prompt 需要优化以处理更多 passages
prompt = f"""
You have {len(passages[:max_passages])} text passages to analyze.

**STRATEGY**:
1. First, quickly scan all passages to identify which ones contain relevant information
2. Focus on passages that mention the CORE ENTITIES: {entities_str}
3. If multiple passages provide the same information, note the consensus
4. If passages contradict each other, note the discrepancy and explain

QUESTION: {task.fact}
CORE ENTITIES: {entities_str}

TEXT PASSAGES:
{combined_text}

**CRITICAL REQUIREMENTS**:
1. Extract information ONLY from the provided passages
2. The information MUST relate to the CORE ENTITIES
3. If multiple passages agree, increase confidence
4. If passages disagree, decrease confidence and explain
5. Quote specific passages (e.g., "Source 3 and Source 7 both state...")

Response format:
{{
    "value": "extracted information",
    "confidence": 0.0-1.0,
    "reasoning": "explanation with source references",
    "source_quotes": ["quote from Source 3", "quote from Source 7"],
    "consensus": "high/medium/low"
}}
"""
```

### 修改 3: 添加 Passage 去重

```python
def _deduplicate_passages(self, passages: List[ContentPassage]) -> List[ContentPassage]:
    """Remove duplicate or highly similar passages"""
    unique_passages = []
    seen_content = set()
    
    for passage in passages:
        # Simple deduplication by content hash
        content_hash = hash(passage.text[:200])  # Use first 200 chars
        
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_passages.append(passage)
    
    return unique_passages
```

---

## Token 消耗分析

### 当前 (7 passages)
```
7 passages × 500 chars/passage = 3,500 chars
3,500 chars ÷ 4 = ~875 tokens
Prompt overhead: ~300 tokens
Total per extraction: ~1,175 tokens
```

### 改进到 20 passages
```
20 passages × 500 chars/passage = 10,000 chars
10,000 chars ÷ 4 = ~2,500 tokens
Prompt overhead: ~300 tokens
Total per extraction: ~2,800 tokens
```

**增加**: 2.4倍 token 消耗

### 改进到 30 passages
```
30 passages × 500 chars/passage = 15,000 chars
15,000 chars ÷ 4 = ~3,750 tokens
Prompt overhead: ~300 tokens
Total per extraction: ~4,050 tokens
```

**增加**: 3.4倍 token 消耗

---

## 成本效益分析

### 场景: 每次查询 3 个 extraction tasks

#### 当前方案 (7 passages)
```
Cost per query: 3 tasks × 1,175 tokens = 3,525 tokens
Accuracy: 70%
Cost per accurate answer: 3,525 / 0.7 = 5,036 tokens
```

#### 方案 A (20 passages)
```
Cost per query: 3 tasks × 2,800 tokens = 8,400 tokens
Accuracy: 85%
Cost per accurate answer: 8,400 / 0.85 = 9,882 tokens
```

**结论**: 虽然 token 增加 2.4倍，但准确性提升 15%，每个准确答案的成本只增加 96%

#### 方案 B (30 passages)
```
Cost per query: 3 tasks × 4,050 tokens = 12,150 tokens
Accuracy: 90%
Cost per accurate answer: 12,150 / 0.9 = 13,500 tokens
```

**结论**: Token 增加 3.4倍，准确性提升 20%，每个准确答案的成本增加 168%

---

## 建议

### 立即实施（今天）

1. **增加 `max_passages_per_task` 到 20**
   ```yaml
   max_passages_per_task: 20
   ```

2. **改进 Extraction Prompt**
   - 添加处理多 passages 的策略
   - 添加一致性检查
   - 添加 `consensus` 字段

3. **添加 Passage 去重**
   - 避免重复内容浪费 tokens

**预期效果**:
- 信息丢失率: 84% → 56%
- 准确性: 70% → 85%
- Token 消耗: +140%

### 本周实施

4. **实现两阶段 Extraction**
   - Stage 1: 快速扫描所有 passages
   - Stage 2: 深度提取候选 passages

5. **添加多源验证**
   - 检查多个 passages 的一致性
   - 标注 consensus level

**预期效果**:
- 信息丢失率: 56% → 0%
- 准确性: 85% → 90%

### 下周实施

6. **升级 Ranking 算法**
   - 实现 semantic similarity ranking
   - 混合 keyword + semantic

**预期效果**:
- 更精准的 ranking
- 即使 passages 数量不变，准确性也提升

---

## 总结

**当前最大的问题**: 7 个 passages 远远不够，信息丢失率 84% 太高

**立即可做的改进**:
1. 增加到 20 个 passages (信息丢失率降到 56%)
2. 改进 prompt 处理更多信息
3. 添加去重避免浪费

**效果**:
- 准确性: 70% → 85% (+15%)
- Token 成本: +140%
- 性价比: 可接受

**下一步**: 实现两阶段 Extraction，达到 90% 准确性

---

最后更新: 2025-10-06
作者: AI Assistant
状态: 📋 方案已制定，待实施

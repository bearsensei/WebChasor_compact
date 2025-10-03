# 多搜索引擎配置指南

WebChaser 支持同时使用多个搜索引擎，合并搜索结果以获得更全面的信息覆盖。

## 配置方法

### 方式 1：单一搜索引擎（默认）

在 `config/config.yaml` 中：

```yaml
ir_rag:
  search:
    provider: serpapi  # 只使用一个搜索引擎
```

可选值：
- `serpapi` - SerpAPI
- `google_custom_search` - Google Custom Search API
- `gcp_vertex_search` - GCP Vertex AI Search

### 方式 2：多个搜索引擎同时使用 ✨

使用**逗号分隔**多个搜索引擎：

```yaml
ir_rag:
  search:
    provider: serpapi,gcp_vertex_search  # 同时使用两个
```

或使用全部三个：

```yaml
ir_rag:
  search:
    provider: serpapi,google_custom_search,gcp_vertex_search
```

## 工作原理

### 单一搜索引擎模式

```
查询 → 生成5个搜索词 → 单一引擎搜索 → 合并去重 → 返回结果
```

示例：
```
[SerpAPI] 5个查询 × 20个结果 = 100个结果 → 去重后 74个唯一结果
```

### 多搜索引擎模式

```
查询 → 生成5个搜索词 → 多个引擎并行搜索 → 合并所有结果 → URL去重 → 返回结果
```

示例：
```
Provider 1 (SerpAPI):        5查询 × 20结果 = 100个结果
Provider 2 (GCP Vertex):     5查询 × 20结果 = 100个结果
───────────────────────────────────────────────
总计:                                   200个结果
去重后:                                 150个唯一结果  (更全面！)
```

## 优势

### 1. **更全面的搜索覆盖**
不同搜索引擎索引的内容有所不同，组合使用可以获得更全面的结果。

### 2. **自动故障转移**
如果某个搜索引擎失败或返回 0 结果，其他引擎的结果仍然可用。

### 3. **结果多样性**
不同引擎的排序算法不同，组合结果提供更多视角。

### 4. **自动去重**
系统自动根据 URL 去重，确保结果唯一性。

## 推荐配置

### 配置 1：成本优化（推荐）
```yaml
provider: serpapi,gcp_vertex_search
```
**说明**：
- SerpAPI 提供丰富的结构化数据
- GCP Vertex 补充更多网络搜索结果
- 避免 Google Custom Search 的严格限制（100次/天）

### 配置 2：最大覆盖
```yaml
provider: serpapi,google_custom_search,gcp_vertex_search
```
**说明**：
- 使用所有三个搜索引擎
- 最全面的搜索覆盖
- 注意 Google Custom Search 的配额限制

### 配置 3：单一高质量
```yaml
provider: gcp_vertex_search
```
**说明**：
- 只使用 GCP Vertex AI Search
- 适合已有自定义索引的场景
- 成本可控，结果质量高

## 性能考虑

### 并发控制

多个搜索引擎会增加总请求数，注意调整并发参数：

```yaml
ir_rag:
  search:
    concurrent: 2   # 降低并发以避免速率限制
    qps: 1          # 每秒查询数
    retries: 2      # 重试次数
```

### 建议设置

| 提供商数量 | concurrent | qps | 说明 |
|-----------|-----------|-----|------|
| 1个 | 8 | 5 | 单一引擎，可以较高并发 |
| 2个 | 2-4 | 1-2 | 避免触发速率限制 |
| 3个 | 2 | 1 | 保守设置，确保稳定性 |

## 日志输出示例

### 单一搜索引擎
```
[IR_RAG][INIT] Using GCP Vertex AI Search
🔍 IR_RAG: Searching with GCP Vertex AI Search...
🔍 IR_RAG: Generated 5 search queries
[GCPVertexSearch] Returned 20 results (×5)
🔍 IR_RAG: Retrieved 74 unique results from 5 queries
```

### 多搜索引擎
```
[IR_RAG][INIT] Using multiple search providers: SerpAPI, GCP Vertex AI Search
🔍 IR_RAG: Searching with multiple providers: SerpAPI, GCP Vertex AI Search...
🔍 IR_RAG: Generated 5 search queries for each provider

🔍 IR_RAG: Searching with serpapi...
[SerpAPI] Returned 20 results (×5)
🔍 IR_RAG: serpapi returned 100 results

🔍 IR_RAG: Searching with gcp_vertex_search...
[GCPVertexSearch] Returned 20 results (×5)
🔍 IR_RAG: gcp_vertex_search returned 100 results

🔍 IR_RAG: Multi-provider search stats: {'serpapi': 100, 'gcp_vertex_search': 100}
🔍 IR_RAG: Total results: 200, Unique results: 150
```

## 环境变量

确保为所有使用的搜索引擎设置相应的环境变量：

```bash
# SerpAPI
export SERPAPI_KEY=your-serpapi-key

# Google Custom Search
export GOOGLE_SEARCH_KEY=your-google-api-key
export GOOGLE_CSE_ID=your-cse-id

# GCP Vertex AI Search
export GCP_PROJECT_ID=your-project-id
export GCP_ENGINE_ID=your-engine-id
export GCP_API_KEY=your-api-key
```

## 故障排查

### 问题 1：某个引擎返回 0 结果

**现象**：
```
🔍 IR_RAG: serpapi returned 0 results
🔍 IR_RAG: gcp_vertex_search returned 100 results
```

**解决方案**：
- 检查该引擎的 API Key 和配置
- 系统会继续使用其他引擎的结果
- 可以暂时从 config 中移除失败的引擎

### 问题 2：速率限制错误

**现象**：
```
[GoogleSearch][ERROR] API Error (429): Too Many Requests
```

**解决方案**：
```yaml
ir_rag:
  search:
    concurrent: 1  # 降低并发
    qps: 0.5       # 降低 QPS（每2秒1个请求）
```

### 问题 3：结果重复过多

**现象**：
```
🔍 IR_RAG: Total results: 200, Unique results: 80
```

**说明**：
- 这是正常的，不同引擎可能返回相同的 URL
- 系统会自动去重
- 重复率高说明搜索引擎覆盖的内容相似

## 最佳实践

### ✅ 推荐做法

1. **从单一引擎开始**
   - 先测试单个引擎是否正常工作
   - 确认 API Key 和配置正确

2. **逐步添加引擎**
   - 确认第一个引擎稳定后再添加第二个
   - 观察日志输出验证结果

3. **调整并发参数**
   - 根据实际速率限制调整 `concurrent` 和 `qps`
   - 避免触发 API 配额限制

4. **监控成本**
   - 多个引擎会增加总请求数和成本
   - 定期检查 API 使用量

### ❌ 避免做法

1. **不要盲目使用全部引擎**
   - 如果预算有限，选择1-2个高质量引擎
   - Google Custom Search 的免费配额很有限（100次/天）

2. **不要设置过高并发**
   - 容易触发速率限制
   - 可能导致请求失败

3. **不要忽略日志警告**
   - 注意 `[WARN]` 和 `[ERROR]` 信息
   - 及时调整配置

## 总结

多搜索引擎配置让 WebChaser 能够：

✅ 获得更全面的搜索结果  
✅ 提高系统可靠性（故障转移）  
✅ 结合不同引擎的优势  
✅ 自动去重和合并结果  

根据你的需求和预算，选择合适的搜索引擎组合！

---

**最后更新**: 2025-10-03


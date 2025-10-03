# GCP Vertex AI Search Integration Summary

## 集成完成

WebChaser 现已成功集成 GCP Vertex AI Search（Google Cloud Discovery Engine）作为第三个搜索提供商选项。

## 修改的文件

### 1. 新增文件
- `src/actions/gcp_vertex_search.py` - GCP Vertex AI Search 模块
  - 统一接口：`call()`, `get_structured_results()`, `batch_call()`, `batch_call_flat()`
  - 支持并发搜索和 QPS 限制
  - 支持重试和错误处理
  - 与现有的 SerpAPI 和 Google Custom Search 接口保持一致

### 2. 修改的核心文件

#### `src/actions/ir_rag.py`
- **第 22 行**：添加 `from actions.gcp_vertex_search import GCPVertexSearch`
- **第 33 行**：在 `RetrievalProvider` 枚举中添加 `GCP_VERTEX_SEARCH = "gcp_vertex_search"`
- **第 744-747 行**：在初始化方法中添加 GCP 搜索工具的初始化逻辑
- **第 840 行**：在 `_retrieve_information()` 方法中添加 GCP 支持
- **第 851-857 行**：更新 `_web_search()` 方法以支持三种搜索提供商

#### `config/config.yaml`
- **第 56 行**：更新注释说明支持的搜索提供商选项

### 3. 更新的文档

#### `docs/SEARCH_PROVIDERS.md`
- 添加 GCP Vertex AI Search 介绍和优缺点
- 添加完整的 GCP 设置步骤（创建项目、启用API、创建搜索引擎、获取凭证）
- 更新功能对比表，包含三个提供商的详细对比
- 添加使用示例和故障排除指南
- 添加成本和推荐说明

## 如何使用

### 方式 1：配置文件切换（推荐）

编辑 `config/config.yaml`:

```yaml
ir_rag:
  search:
    provider: gcp_vertex_search  # 选项: 'serpapi', 'google_custom_search', 'gcp_vertex_search'
    qps: 5                       # 每秒查询数
    concurrent: 8                # 并发请求数
    retries: 2                   # 重试次数
```

### 方式 2：代码中切换

```python
from actions.ir_rag import IR_RAG, IRConfig, RetrievalProvider

# 使用 GCP Vertex AI Search
config = IRConfig(
    search_provider=RetrievalProvider.GCP_VERTEX_SEARCH
)
ir_rag = IR_RAG(config=config, llm_client=client)
```

## 环境变量配置

使用 GCP Vertex AI Search 需要设置以下环境变量：

```bash
export GCP_PROJECT_ID=your-gcp-project-id
export GCP_ENGINE_ID=your-vertex-search-engine-id
export GCP_API_KEY=your-gcp-api-key
export GCP_LOCATION=global  # 可选，默认值: global
```

## 依赖安装

如果使用 GCP Vertex AI Search，需要安装额外的依赖：

```bash
pip install google-cloud-discoveryengine
```

## 测试

测试单个搜索：

```bash
# 设置环境变量
export GCP_PROJECT_ID=your-project-id
export GCP_ENGINE_ID=your-engine-id
export GCP_API_KEY=your-api-key

# 运行测试
python src/actions/gcp_vertex_search.py
```

测试集成到 IR_RAG：

```bash
# 修改 config.yaml 设置 provider: gcp_vertex_search
# 然后运行
python src/main.py
```

## 三种搜索提供商对比

| 特性 | SerpAPI | Google Custom Search | GCP Vertex AI Search |
|------|---------|---------------------|---------------------|
| 每次请求结果数 | 最多 100 | 最多 10 | 可配置 |
| 知识图谱 | ✓ | ✗ | ✗ |
| 答案框 | ✓ | ✗ | ✗ |
| 自定义数据源 | ✗ | ✗ | ✓ |
| 相关性调优 | ✗ | ✗ | ✓ |
| 免费额度 | 100次/月 | 100次/天 | 有限试用 |
| 付费价格 | $50/5K次 | $5/1K次 | ~$0.01/次 |

## 推荐使用场景

### 使用 SerpAPI 如果：
- 需要丰富的结构化数据（知识图谱等）
- 每次查询需要超过 10 个结果
- 预算允许（$50+/月）
- 需要更好的速率限制

### 使用 Google Custom Search 如果：
- 预算紧张（每天 100 次免费搜索）
- 每次查询 10 个结果足够
- 偏好官方 Google API
- 查询量较低到中等

### 使用 GCP Vertex AI Search 如果：
- 需要企业级 AI 搜索能力
- 需要索引自定义数据源
- 需要高级相关性调优和个性化
- 已有 GCP 基础设施
- 预算允许按查询计费

## 架构说明

系统采用统一的接口设计，所有搜索提供商都实现相同的方法：

1. **`call(params)`** - 基本搜索接口
2. **`get_structured_results()`** - 返回结构化搜索结果
3. **`batch_call()`** - 批量并发搜索
4. **`batch_call_flat()`** - 批量搜索并展平结果

这种设计使得切换搜索提供商非常简单，只需修改配置文件即可，无需改动业务逻辑代码。

## 未来扩展

系统架构预留了混合搜索（HYBRID）模式的支持，未来可以实现：
- 多个搜索提供商同时使用
- 自动故障转移
- 结果融合和去重
- 智能提供商选择

## 总结

✅ GCP Vertex AI Search 已完全集成到 WebChaser 系统
✅ 与现有的 SerpAPI 和 Google Custom Search 无缝共存
✅ 统一的接口设计，易于切换和扩展
✅ 完整的文档和使用指南
✅ 支持批量搜索和并发控制

---
**集成完成日期**: 2025-10-03


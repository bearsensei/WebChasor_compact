# Google Custom Search API 配额限制说明

## 问题现象

当您看到以下错误时：
```
429 Client Error: Too Many Requests
```

说明您遇到了 Google Custom Search API 的速率限制。

## Google API 配额限制

### 免费版限制
- **每日查询**: 100 次/天
- **QPS（每秒查询）**: ~1 次/秒
- **并发请求**: 建议不超过 2 个

### 付费版限制
- **每日查询**: 10,000 次/天（$5/1000 次）
- **QPS**: 仍然较低（~10 次/秒）
- **并发请求**: 建议不超过 5 个

## 解决方案

### 方案 1：调整配置（当前方案）

编辑 `config/config.yaml`：

```yaml
ir_rag:
  search:
    provider: google_custom_search
    concurrent: 2       # 降低并发（默认 8 → 2）
    qps: 1             # 降低 QPS（默认 5 → 1）
    retries: 2         # 保持重试次数
```

**优点**: 继续使用 Google API
**缺点**: 搜索速度较慢，配额仍然有限

---

### 方案 2：切换到 SerpAPI（推荐）

SerpAPI 提供更好的速率限制和更丰富的数据。

#### 步骤 1: 获取 SerpAPI Key
1. 访问 [https://serpapi.com/](https://serpapi.com/)
2. 注册账号
3. 复制 API Key

#### 步骤 2: 配置环境变量
```bash
# .env
SERPAPI_KEY=your-serpapi-key-here
```

#### 步骤 3: 修改配置
```yaml
# config/config.yaml
ir_rag:
  search:
    provider: serpapi     # 改为 serpapi
    concurrent: 8         # 可以提高并发
    qps: 5               # 可以提高 QPS
```

**优点**: 
- 更高的速率限制
- 更丰富的结构化数据（知识图谱、答案框等）
- 更好的稳定性

**缺点**: 
- 需要付费订阅（$50/月起）

---

### 方案 3：混合使用

为不同场景使用不同的搜索引擎：

**开发/测试**: Google Custom Search（免费）
```yaml
provider: google_custom_search
concurrent: 2
qps: 1
```

**生产环境**: SerpAPI（付费，更稳定）
```yaml
provider: serpapi
concurrent: 8
qps: 5
```

---

## 当前已实施的优化

系统已经实施了以下优化来应对 Google API 限制：

### 1. 降低速率
- **并发数**: 8 → 2
- **QPS**: 5 → 1

### 2. 增加重试延迟
- 初始延迟: 0.6s → 2.0s
- 遇到 429 错误时延迟加倍

### 3. 智能重试
```python
# 检测到 429 错误时使用更长的延迟
if "429" in error_msg or "Too Many Requests" in error_msg:
    sleep_time = backoff * 2 * (1.0 + random.random() * 0.5)
```

### 4. 友好提示
系统会在遇到配额限制时提示：
```
[GoogleSearch][HINT] Google Custom Search has strict rate limits (1 QPS, 100/day free).
[GoogleSearch][HINT] Consider switching to 'serpapi' in config.yaml for better limits.
```

---

## 配额计算示例

### 场景 1: 单次查询
- **查询数**: 1 个原始查询
- **多样化查询**: QueryMaker 生成 5 个查询
- **总消耗**: 5 次 API 调用
- **免费配额**: 100/5 = 20 次这样的查询/天

### 场景 2: 批量查询
- **查询数**: 3 个原始查询
- **多样化查询**: 每个生成 5 个 = 15 个查询
- **总消耗**: 15 次 API 调用
- **免费配额**: 100/15 = 6 次这样的批量查询/天

---

## 监控配额使用

### 查看 Google API 配额
1. 访问 [Google Cloud Console](https://console.cloud.google.com/)
2. 选择您的项目
3. 导航到 "APIs & Services" → "Dashboard"
4. 点击 "Custom Search API"
5. 查看 "Quotas" 标签

### 实时监控
在运行时，系统会打印：
```
[GoogleSearch] Searching: 'query' (location: Hong Kong/hk, ...)
[GoogleSearch] Returned 10 results
```

如果看到：
```
[GoogleSearch][ERROR] API Error (429): Rate limit exceeded
[GoogleSearch] Rate limit hit, waiting 4.5s before retry 1/2
```

说明遇到了速率限制。

---

## 推荐配置对比

| 配置项 | Google Free | Google Paid | SerpAPI |
|--------|-------------|-------------|---------|
| **QPS** | 1 | 5 | 10+ |
| **并发** | 2 | 5 | 20+ |
| **日限额** | 100 | 10,000 | 5,000+ |
| **成本** | 免费 | $5/1K | $50/5K |
| **数据丰富度** | 基础 | 基础 | 丰富 |
| **稳定性** | 一般 | 好 | 很好 |

---

## 故障排除

### 问题 1: 仍然遇到 429 错误

**原因**: QPS 设置仍然太高

**解决**:
```yaml
qps: 0.5  # 进一步降低到 0.5 次/秒
```

---

### 问题 2: 每日配额用完

**解决方案**:
1. **等待到第二天**（UTC 时区重置）
2. **升级到付费版**
3. **切换到 SerpAPI**

---

### 问题 3: 搜索太慢

**原因**: 降低 QPS 后，5 个查询需要 5 秒

**解决**:
1. **减少多样化查询数量**:
   ```yaml
   models:
     querymaker:
       num_queries: 3  # 从 5 降到 3
   ```

2. **或切换到 SerpAPI**（更快）

---

## 测试当前配置

运行测试脚本验证配置：

```bash
# 单次测试（消耗 1 次配额）
python -c "
from actions.google_search import GoogleCustomSearch
search = GoogleCustomSearch()
results = search.get_structured_results('测试查询', num_results=5)
print(f'Got {len(results)} results')
"

# 检查是否遇到速率限制
# 如果成功 → 配置合理
# 如果 429 → 需要进一步降低 QPS
```

---

## 最佳实践建议

### 对于开发者
- **开发/调试**: 使用 Google Custom Search（免费）
- **设置**: `qps: 1, concurrent: 2`
- **注意**: 控制测试频率，避免用完配额

### 对于生产环境
- **推荐**: 使用 SerpAPI
- **备选**: Google Custom Search 付费版
- **原因**: 
  - 更稳定
  - 速率限制更宽松
  - 数据更丰富
  - 用户体验更好

---

## 总结

**当前状态**: 
- ✅ 已优化配置以适应 Google API 限制
- ✅ QPS: 1, 并发: 2
- ✅ 增强的重试机制
- ✅ 友好的错误提示

**如果仍遇到问题**:
1. 检查配额使用情况
2. 等待配额重置
3. 考虑切换到 SerpAPI

**长期建议**:
- 生产环境使用 SerpAPI
- Google API 用于开发/测试

---

**最后更新**: 2025-01-10


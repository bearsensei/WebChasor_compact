# WebChasor Configuration Guide

## Configuration Priority System

WebChasor 支持**三级配置优先级系统**，灵活适配不同部署场景：

```
┌─────────────────────────────────────────┐
│    优先级 1: 环境变量 (ENV)              │  ← 最高优先级
├─────────────────────────────────────────┤
│    优先级 2: config.yaml                │  ← 中等优先级
├─────────────────────────────────────────┤
│    优先级 3: 代码默认值                  │  ← 最低优先级
└─────────────────────────────────────────┘
```

---

## Why Three Levels?

### Level 1: 环境变量 (Highest Priority)

**使用场景**:
- ✅ Production deployments
- ✅ Docker / Kubernetes
- ✅ Different environments (dev/test/prod)
- ✅ Sensitive information (API keys)

**优点**:
- 无需重新构建镜像即可修改配置
- 容器编排工具天然支持
- 敏感信息不会进入代码仓库
- 符合 12-Factor App 标准

**示例**:
```bash
export WEB_WORKERS=16
export WORKER_CONNECTIONS=2000
export OPENAI_API_KEY_AGENT="sk-xxx"
```

---

### Level 2: config.yaml (Middle Priority)

**使用场景**:
- ✅ Development environment
- ✅ Default configurations
- ✅ Team-wide settings
- ✅ Version controlled settings

**优点**:
- 集中管理所有配置
- 版本控制，可追溯
- 开发者友好，易于查看和修改
- 支持复杂的嵌套结构

**示例**:
```yaml
# config/config.yaml
performance:
  gunicorn:
    workers: 8
    worker_connections: 1000
    worker_timeout: 120
```

---

### Level 3: 代码默认值 (Lowest Priority)

**使用场景**:
- ✅ Fallback values
- ✅ Ensures system can start
- ✅ Reasonable defaults for all scenarios

**优点**:
- 保证系统即使没有配置也能启动
- 提供经过验证的默认值
- 减少配置负担

---

## Gunicorn Server Configuration

### Available Parameters

| Parameter | ENV Variable | YAML Key | Default | Description |
|-----------|-------------|----------|---------|-------------|
| Workers | `WEB_WORKERS` | `workers` | CPU×2+1 | Worker进程数 |
| Connections | `WORKER_CONNECTIONS` | `worker_connections` | 1000 | 每个worker连接数 |
| Timeout | `WORKER_TIMEOUT` | `worker_timeout` | 120 | Worker超时（秒） |
| Backlog | `BACKLOG` | `backlog` | 2048 | 挂起连接队列 |
| Max Requests | `MAX_REQUESTS` | `max_requests` | 1000 | Worker重启阈值 |
| Jitter | `MAX_REQUESTS_JITTER` | `max_requests_jitter` | 50 | 重启随机抖动 |
| Keepalive | `KEEPALIVE` | `keepalive` | 5 | Keep-alive超时 |
| Log Level | `LOG_LEVEL` | `log_level` | info | 日志级别 |

---

## Usage Examples

### Example 1: Development (Using YAML)

**config/config.yaml**:
```yaml
performance:
  gunicorn:
    workers: 4            # 少量worker，方便调试
    worker_connections: 500
    worker_timeout: 120
    log_level: "debug"    # 详细日志
```

**Start server**:
```bash
gunicorn -c src/api/gunicorn.conf.py src.api.api_server:app
```

**Result**: 使用 YAML 中的配置

---

### Example 2: Production (Using ENV)

**Set environment variables**:
```bash
export WEB_WORKERS=16
export WORKER_CONNECTIONS=2000
export WORKER_TIMEOUT=180
export LOG_LEVEL=warning
```

**Start server**:
```bash
gunicorn -c src/api/gunicorn.conf.py src.api.api_server:app
```

**Result**: 环境变量覆盖 YAML 配置

---

### Example 3: Docker (ENV + YAML)

**config/config.yaml** (Default values for all environments):
```yaml
performance:
  gunicorn:
    workers: null         # null = auto-calculate
    worker_connections: 1000
    worker_timeout: 120
    log_level: "info"
```

**Dockerfile**:
```dockerfile
COPY config/ /app/config/
# YAML 配置在镜像中
```

**docker-compose.yml** (Override for production):
```yaml
version: '3.8'
services:
  api:
    image: webchasor-api:latest
    environment:
      - WEB_WORKERS=16          # 覆盖 YAML
      - WORKER_CONNECTIONS=2000  # 覆盖 YAML
      - OPENAI_API_KEY_AGENT=${API_KEY}  # 敏感信息
```

**Result**: 
- 镜像包含 YAML 默认配置
- 运行时用环境变量覆盖特定值
- 无需重新构建镜像即可调整配置

---

### Example 4: Kubernetes (ConfigMap + Secret)

**ConfigMap** (Non-sensitive config):
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: webchasor-config
data:
  config.yaml: |
    performance:
      gunicorn:
        workers: 8
        worker_connections: 1000
```

**Deployment** (Override with ENV):
```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: api
        env:
        # 覆盖特定环境的值
        - name: WEB_WORKERS
          value: "16"
        - name: WORKER_CONNECTIONS
          value: "2000"
        # 敏感信息从 Secret 读取
        - name: OPENAI_API_KEY_AGENT
          valueFrom:
            secretKeyRef:
              name: webchasor-secrets
              key: openai-api-key
        volumeMounts:
        - name: config
          mountPath: /app/config
      volumes:
      - name: config
        configMap:
          name: webchasor-config
```

**Result**:
- ConfigMap 提供基础配置
- ENV 覆盖特定值
- Secret 管理敏感信息

---

## Application-Level Configuration

### Semaphore Limits

**config.yaml**:
```yaml
performance:
  concurrency:
    max_concurrent_requests: 100  # 全局并发限制
    llm_concurrent_limit: 20      # LLM API 并发限制
```

**说明**: 这些限制是**应用级别**的，与 Gunicorn 的 worker 并发控制**协同工作**。

---

## Best Practices

### 1. 环境分离 (Environment Separation)

```bash
# Development
export WEB_WORKERS=4
export LOG_LEVEL=debug

# Production
export WEB_WORKERS=16
export LOG_LEVEL=warning
```

### 2. 敏感信息管理 (Secrets Management)

**❌ 不要这样做**:
```yaml
# config.yaml
openai:
  api_key: "sk-xxxxx"  # 不要把密钥写在 YAML 里！
```

**✅ 正确做法**:
```bash
# Use environment variables for secrets
export OPENAI_API_KEY_AGENT="sk-xxxxx"
```

### 3. 默认值设置 (Default Values)

**config.yaml** 中设置**合理的默认值**:
```yaml
performance:
  gunicorn:
    workers: null         # auto-calculate
    worker_connections: 1000
    worker_timeout: 120
```

### 4. 文档化配置 (Document Configuration)

在 `config.yaml` 中添加注释说明每个参数的作用。

---

## Configuration Validation

### How to Check Current Config

启动服务时，Gunicorn 会打印当前使用的配置：

```
[GUNICORN] ========================================
[GUNICORN] Configuration Priority: ENV > YAML > DEFAULT
[GUNICORN] ========================================
[GUNICORN] Workers: 16
[GUNICORN] Worker connections: 2000
[GUNICORN] Worker timeout: 180s
[GUNICORN] Backlog: 4096
[GUNICORN] Max requests: 1000
[GUNICORN] Log level: info
[GUNICORN] ========================================
[GUNICORN] Total capacity: ~32000 concurrent connections
[GUNICORN] ========================================
```

### Verification Checklist

- [ ] Workers 数量合理（不要超过 CPU 核心数 × 4）
- [ ] Total capacity 满足预期负载
- [ ] Timeout 足够长（LLM 查询可能需要 120-180 秒）
- [ ] Log level 适合环境（dev=debug, prod=warning）
- [ ] 敏感信息通过环境变量传递，未出现在日志中

---

## Troubleshooting

### Problem: Config Not Taking Effect

**Check priority**:
1. 环境变量是否设置？`echo $WEB_WORKERS`
2. YAML 文件是否存在？`cat config/config.yaml`
3. 启动日志中显示的配置值是什么？

### Problem: Can't Load config.yaml

**Error**:
```
[GUNICORN] Warning: Could not load config.yaml: ...
[GUNICORN] Using default values and environment variables
```

**Solutions**:
- 检查 `config/config.yaml` 文件是否存在
- 检查 YAML 语法是否正确
- 确保 Python path 包含项目根目录

---

## Migration Guide

### 从纯环境变量迁移到混合模式

**Before**:
```bash
# 所有配置都在启动脚本中
export WEB_WORKERS=8
export WORKER_CONNECTIONS=1000
export WORKER_TIMEOUT=120
export LOG_LEVEL=info
...
gunicorn -c src/api/gunicorn.conf.py src.api.api_server:app
```

**After**:
```yaml
# config.yaml - 默认配置
performance:
  gunicorn:
    workers: 8
    worker_connections: 1000
    worker_timeout: 120
    log_level: "info"
```

```bash
# 启动脚本 - 只需要覆盖特定值
export WEB_WORKERS=16  # 只覆盖需要改的
gunicorn -c src/api/gunicorn.conf.py src.api.api_server:app
```

---

## Summary

### Configuration Priority

```
环境变量 (ENV) > config.yaml > 代码默认值
```

### When to Use What

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 本地开发 | YAML | 方便查看和修改 |
| 生产部署 | ENV + YAML | YAML 提供默认值，ENV 覆盖特定值 |
| 敏感信息 | ENV only | 不进入代码仓库 |
| 多环境 | ENV + YAML | YAML 共享配置，ENV 差异化配置 |
| Docker/K8s | ENV + YAML | 最灵活的组合 |

### Quick Reference

**需要修改配置时**:

1. **临时测试**: 直接 `export ENV_VAR=value`
2. **开发默认值**: 修改 `config/config.yaml`
3. **生产部署**: 在 Kubernetes/Docker 中设置环境变量

**配置生效顺序**:

```
ENV有值 → 使用ENV
  ↓ 没有
YAML有值 → 使用YAML
  ↓ 没有
使用代码默认值
``` 
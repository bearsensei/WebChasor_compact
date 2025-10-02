# WebChasor API Performance Tuning Guide

## Overview

This guide explains how to optimize WebChasor API for high concurrency without modifying code.

---

## Architecture

### Multi-Level Concurrency Control

```
┌─────────────────────────────────────────────────┐
│           Load Balancer (Optional)              │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼──────┐            ┌───────▼──────┐
│  Gunicorn    │            │  Gunicorn    │
│  Worker 1    │    ...     │  Worker N    │
│              │            │              │
│ ┌──────────┐ │            │ ┌──────────┐ │
│ │ Uvicorn  │ │            │ │ Uvicorn  │ │
│ │ (Async)  │ │            │ │ (Async)  │ │
│ └──────────┘ │            │ └──────────┘ │
│              │            │              │
│ Semaphores:  │            │ Semaphores:  │
│ - Global:100 │            │ - Global:100 │
│ - LLM: 20    │            │ - LLM: 20    │
└──────────────┘            └──────────────┘
```

**3层并发控制**:
1. **Gunicorn Workers** - 进程级并发
2. **Uvicorn Async** - 协程级并发  
3. **Semaphores** - 资源限流

---

## Gunicorn Configuration Parameters

### Core Parameters

#### 1. `workers` (Worker 进程数)

**Default**: `CPU cores * 2 + 1`

```bash
# 环境变量设置
export WEB_WORKERS=8
```

**推荐配置**:
- **CPU 密集型**: `CPU cores + 1`
- **I/O 密集型**: `CPU cores * 2 + 1` ✅ (LLM API 属于此类)
- **超高并发**: `CPU cores * 4`

**示例**:
- 4核机器 → 默认 9 workers
- 8核机器 → 默认 17 workers
- 16核机器 → 默认 33 workers

#### 2. `worker_connections` (每个 Worker 的连接数)

**Default**: `1000`

```bash
export WORKER_CONNECTIONS=2000
```

**总并发能力** = `workers * worker_connections`

**示例**:
- 9 workers × 1000 = 9,000 并发连接
- 9 workers × 2000 = 18,000 并发连接

#### 3. `timeout` (Worker 超时)

**Default**: `120` 秒

```bash
export WORKER_TIMEOUT=180
```

**推荐值**:
- 快速查询: 60秒
- 标准查询: 120秒 ✅
- 长时间查询: 180-300秒

#### 4. `backlog` (挂起连接队列)

**Default**: `2048`

```bash
export BACKLOG=4096
```

**说明**: 在所有 workers 都忙时，系统可以排队的最大连接数

---

## Performance Tuning Scenarios

### Scenario 1: 低流量生产环境 (< 100 QPS)

```bash
export WEB_WORKERS=4
export WORKER_CONNECTIONS=500
export WORKER_TIMEOUT=120
```

**容量**: ~2,000 并发连接

---

### Scenario 2: 中等流量 (100-500 QPS)

```bash
export WEB_WORKERS=8
export WORKER_CONNECTIONS=1000
export WORKER_TIMEOUT=120
```

**容量**: ~8,000 并发连接

---

### Scenario 3: 高流量 (500-2000 QPS)

```bash
export WEB_WORKERS=16
export WORKER_CONNECTIONS=2000
export WORKER_TIMEOUT=120
export BACKLOG=4096
```

**容量**: ~32,000 并发连接

---

### Scenario 4: 超高流量 (> 2000 QPS)

```bash
export WEB_WORKERS=32
export WORKER_CONNECTIONS=2000
export WORKER_TIMEOUT=120
export BACKLOG=8192
```

**容量**: ~64,000 并发连接

**额外建议**:
- 使用负载均衡器（Nginx, HAProxy）
- 部署多个实例
- 启用 Redis 缓存

---

## Application-Level Concurrency Control

### Semaphore Configuration

在 `config/config.yaml` 中配置:

```yaml
performance:
  concurrency:
    max_concurrent_requests: 100  # Global concurrent request limit
    llm_concurrent_limit: 20      # LLM concurrent request limit
```

**说明**:
- `max_concurrent_requests`: 全局并发限制（跨所有请求类型）
- `llm_concurrent_limit`: LLM API 调用限制（避免超过 API 速率限制）

**推荐配置**:

| 环境 | max_concurrent_requests | llm_concurrent_limit |
|------|-------------------------|---------------------|
| Dev  | 50                      | 10                  |
| Test | 100                     | 20                  |
| Prod (小) | 200                | 50                  |
| Prod (大) | 500                | 100                 |

---

## Monitoring & Metrics

### Key Metrics to Monitor

1. **Worker Utilization**
   - 监控活跃 worker 数量
   - 如果经常满载，增加 workers

2. **Response Time**
   - P50, P95, P99 延迟
   - 如果 P99 > 10s，检查瓶颈

3. **Error Rate**
   - 超时错误 (503)
   - 如果 > 1%，增加 timeout 或 workers

4. **Memory Usage**
   - 每个 worker ~200-500MB
   - 总内存 = workers × per_worker_memory

### Gunicorn Stats

```bash
# 查看 Gunicorn 状态
ps aux | grep gunicorn

# 查看连接数
netstat -an | grep :8000 | wc -l
```

---

## Deployment Examples

### Docker Compose

```yaml
version: '3.8'
services:
  webchasor-api:
    image: webchasor-api:latest
    environment:
      - WEB_WORKERS=8
      - WORKER_CONNECTIONS=1000
      - WORKER_TIMEOUT=120
      - MAX_CONCURRENT_REQUESTS=200
      - LLM_CONCURRENT_LIMIT=50
    ports:
      - "8000:8000"
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webchasor-api
spec:
  replicas: 3  # 3个 Pod
  template:
    spec:
      containers:
      - name: api
        image: webchasor-api:latest
        env:
        - name: WEB_WORKERS
          value: "8"
        - name: WORKER_CONNECTIONS
          value: "1000"
        resources:
          requests:
            cpu: "2000m"
            memory: "4Gi"
          limits:
            cpu: "4000m"
            memory: "8Gi"
```

**总容量**: 3 pods × 8 workers × 1000 = 24,000 并发连接

---

## Troubleshooting

### Problem: Workers Timeout Frequently

**Symptoms**: 
- Logs show "Worker timeout (pid:xxx)"
- 503 errors

**Solutions**:
1. Increase `WORKER_TIMEOUT`:
   ```bash
   export WORKER_TIMEOUT=180
   ```
2. Check LLM API response time
3. Add more workers

### Problem: High Memory Usage

**Symptoms**:
- OOM kills
- Slow response times

**Solutions**:
1. Reduce workers:
   ```bash
   export WEB_WORKERS=4
   ```
2. Enable worker recycling (already configured):
   ```python
   max_requests = 1000
   max_requests_jitter = 50
   ```

### Problem: Connection Refused

**Symptoms**:
- Client sees "Connection refused"
- Logs show "backlog full"

**Solutions**:
1. Increase backlog:
   ```bash
   export BACKLOG=4096
   ```
2. Add more workers
3. Add load balancer

---

## Best Practices

### 1. Start Conservative, Scale Up

```bash
# Start with
export WEB_WORKERS=4

# Monitor and increase if needed
export WEB_WORKERS=8
```

### 2. Monitor Before Tuning

Use metrics to drive decisions, not guesses.

### 3. Test Under Load

```bash
# Use wrk or ab for load testing
wrk -t12 -c400 -d30s http://localhost:8000/health
```

### 4. Consider Resource Limits

| Resource | Calculation |
|----------|-------------|
| CPU | ≥ workers / 2 cores |
| Memory | workers × 500MB |
| File Descriptors | workers × connections |

### 5. Use Horizontal Scaling

Instead of 1 server with 32 workers:
- Deploy 4 servers with 8 workers each
- Better fault tolerance
- Easier to scale

---

## Quick Reference

| Parameter | Environment Variable | Default | Recommended Range |
|-----------|---------------------|---------|-------------------|
| Workers | `WEB_WORKERS` | CPU×2+1 | 4-32 |
| Connections | `WORKER_CONNECTIONS` | 1000 | 500-2000 |
| Timeout | `WORKER_TIMEOUT` | 120s | 60-300s |
| Backlog | `BACKLOG` | 2048 | 2048-8192 |
| Max Requests | `MAX_REQUESTS` | 1000 | 500-2000 |
| Log Level | `LOG_LEVEL` | info | debug/info/warning |

---

## Summary

**无需修改代码**，只需调整环境变量即可优化性能：

```bash
# 高并发配置示例
export WEB_WORKERS=16
export WORKER_CONNECTIONS=2000
export WORKER_TIMEOUT=120
export BACKLOG=4096

# 启动服务
gunicorn -c src/api/gunicorn.conf.py src.api.api_server:app
```

**Expected Performance**:
- 16 workers × 2000 connections = 32,000 并发
- 配合应用层 Semaphore，稳定支持高并发场景 
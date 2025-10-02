# gunicorn.conf.py
# Gunicorn 配置文件 - 高并发优化版
# 支持优先级: 环境变量 > config.yaml > 代码默认值
import multiprocessing
import os
import sys

# Add project root to path to import config_manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Try to load config from YAML, fallback to defaults if not available
try:
    from config_manager import get_config
    cfg = get_config()
    gunicorn_cfg = cfg.get('performance.gunicorn', {})
except Exception as e:
    print(f"[GUNICORN] Warning: Could not load config.yaml: {e}")
    print(f"[GUNICORN] Using default values and environment variables")
    gunicorn_cfg = {}

# Helper function to get config value with priority: ENV > YAML > DEFAULT
def get_config_value(env_var, yaml_key, default):
    """Get config value with priority: ENV > YAML > DEFAULT"""
    # 1. Try environment variable first
    env_value = os.getenv(env_var)
    if env_value is not None:
        return int(env_value) if isinstance(default, int) else env_value
    
    # 2. Try YAML config
    yaml_value = gunicorn_cfg.get(yaml_key)
    if yaml_value is not None:
        return yaml_value
    
    # 3. Use default
    return default

# ============================================================================
# 服务器配置
# ============================================================================

# 绑定地址和端口
bind = "0.0.0.0:8000"

# Worker 进程数 (建议：CPU 核心数 * 2 + 1)
# 优先级: WEB_WORKERS (env) > config.yaml > auto-calculate
_workers_cfg = get_config_value("WEB_WORKERS", "workers", None)
workers = _workers_cfg if _workers_cfg else multiprocessing.cpu_count() * 2 + 1

# Worker 类型：使用 Uvicorn 的异步 worker
worker_class = "uvicorn.workers.UvicornWorker"

# 每个 worker 的最大并发连接数
# 优先级: WORKER_CONNECTIONS (env) > config.yaml > 1000
worker_connections = get_config_value("WORKER_CONNECTIONS", "worker_connections", 1000)

# Worker 线程数 (对于异步 worker，通常设为 1)
threads = 1

# ============================================================================
# 日志配置
# ============================================================================

# 访问日志输出到 stdout
accesslog = "-"

# 错误日志输出到 stderr
errorlog = "-"

# 日志级别: debug, info, warning, error, critical
# 优先级: LOG_LEVEL (env) > config.yaml > "info"
loglevel = get_config_value("LOG_LEVEL", "log_level", "info")

# 访问日志格式
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# ============================================================================
# 进程配置
# ============================================================================

# 预加载应用（节省内存，加快启动）
preload_app = True

# Worker 重启前处理的最大请求数（防止内存泄漏）
# 优先级: MAX_REQUESTS (env) > config.yaml > 1000
max_requests = get_config_value("MAX_REQUESTS", "max_requests", 1000)

# 随机抖动，避免所有 worker 同时重启
# 优先级: MAX_REQUESTS_JITTER (env) > config.yaml > 50
max_requests_jitter = get_config_value("MAX_REQUESTS_JITTER", "max_requests_jitter", 50)

# Graceful 重启超时时间（秒）
graceful_timeout = 30

# ============================================================================
# 超时配置
# ============================================================================

# Worker 超时时间（秒）- 处理单个请求的最大时间
# 对于 LLM 应用，需要较长的超时时间
# 优先级: WORKER_TIMEOUT (env) > config.yaml > 120
timeout = get_config_value("WORKER_TIMEOUT", "worker_timeout", 120)

# Keep-alive 连接超时（秒）
# 优先级: KEEPALIVE (env) > config.yaml > 5
keepalive = get_config_value("KEEPALIVE", "keepalive", 5)

# ============================================================================
# 性能优化
# ============================================================================

# 使用 eventlet 或 gevent 实现异步（对于 uvicorn worker，这个不需要）
# worker_class 已经是 uvicorn.workers.UvicornWorker，天然支持异步

# 最大挂起连接数
# 优先级: BACKLOG (env) > config.yaml > 2048
backlog = get_config_value("BACKLOG", "backlog", 2048)

# ============================================================================
# 安全配置
# ============================================================================

# Worker 运行用户和组（Docker 容器中由 Dockerfile 设置，这里不覆盖）
user = None
group = None

# 临时上传目录
tmp_upload_dir = None

# 限制请求行大小（bytes）
limit_request_line = 4096

# 限制请求头字段数量
limit_request_fields = 100

# 限制请求头大小（bytes）
limit_request_field_size = 8190

# ============================================================================
# 钩子函数（可选）
# ============================================================================

def on_starting(server):
    """服务器启动前"""
    print(f"[GUNICORN] ========================================")
    print(f"[GUNICORN] Configuration Priority: ENV > YAML > DEFAULT")
    print(f"[GUNICORN] ========================================")
    print(f"[GUNICORN] Workers: {workers}")
    print(f"[GUNICORN] Worker connections: {worker_connections}")
    print(f"[GUNICORN] Worker timeout: {timeout}s")
    print(f"[GUNICORN] Backlog: {backlog}")
    print(f"[GUNICORN] Max requests: {max_requests}")
    print(f"[GUNICORN] Log level: {loglevel}")
    print(f"[GUNICORN] ========================================")
    print(f"[GUNICORN] Total capacity: ~{workers * worker_connections} concurrent connections")
    print(f"[GUNICORN] ========================================")

def when_ready(server):
    """服务器准备就绪"""
    print(f"[GUNICORN] Server is ready. Listening on: {bind}")

def on_exit(server):
    """服务器退出时"""
    print("[GUNICORN] Server shutting down")

# ============================================================================
# 配置优先级说明
# ============================================================================
# 所有配置项支持三种来源，优先级从高到低：
#
# 1. 环境变量 (ENV) - 最高优先级
#    - 适用场景：Docker, Kubernetes, 不同环境的部署
#    - 优点：灵活、安全、无需重新构建镜像
#
# 2. config.yaml - 中等优先级
#    - 适用场景：开发环境、默认配置
#    - 配置路径：config/config.yaml -> performance.gunicorn
#    - 优点：集中管理、版本控制
#
# 3. 代码默认值 - 最低优先级
#    - 适用场景：兜底配置
#    - 优点：保证系统能够启动
#
# ============================================================================
# 可配置参数列表
# ============================================================================
#
# 参数名 (ENV变量 / YAML键 / 默认值)
#
# WEB_WORKERS / workers / CPU×2+1
#   Worker 进程数
#
# WORKER_CONNECTIONS / worker_connections / 1000
#   每个 worker 的最大连接数
#
# WORKER_TIMEOUT / worker_timeout / 120
#   Worker 超时时间（秒）
#
# BACKLOG / backlog / 2048
#   最大挂起连接数
#
# MAX_REQUESTS / max_requests / 1000
#   Worker 处理多少请求后重启
#
# MAX_REQUESTS_JITTER / max_requests_jitter / 50
#   Worker 重启随机抖动
#
# KEEPALIVE / keepalive / 5
#   Keep-alive 连接超时（秒）
#
# LOG_LEVEL / log_level / info
#   日志级别 (debug, info, warning, error, critical)
#
# ============================================================================
# 使用示例
# ============================================================================
#
# 方式1: 使用环境变量（推荐用于生产环境）
#   export WEB_WORKERS=16
#   export WORKER_CONNECTIONS=2000
#   gunicorn -c src/api/gunicorn.conf.py src.api.api_server:app
#
# 方式2: 修改 config.yaml（推荐用于开发环境）
#   编辑 config/config.yaml:
#   performance:
#     gunicorn:
#       workers: 8
#       worker_connections: 1000
#
# 方式3: 混合使用（推荐）
#   在 config.yaml 中设置默认值
#   用环境变量覆盖特定环境的配置
#
# ============================================================================
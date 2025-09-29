# gunicorn.conf.py
# Gunicorn 配置文件
import multiprocessing
import os

# 服务器配置
bind = "0.0.0.0:8000"
workers = 1  # 由于使用了全局状态，建议使用单个worker
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# 日志配置
accesslog = "-"  # 输出到stdout
errorlog = "-"   # 输出到stderr
loglevel = "info"

# 进程配置
preload_app = True
max_requests = 1000
max_requests_jitter = 50

# 超时配置
timeout = 120
keepalive = 5

# 其他配置
user = None
group = None
tmp_upload_dir = None
#!/bin/bash
# 启动 WebChasor API 服务器的脚本

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# 进入项目目录
cd "$PROJECT_ROOT"

# 尝试找到并激活虚拟环境
VENV_PATHS=(
    "$HOME/miniconda3/envs/webchaser/bin/activate"
    "$HOME/anaconda3/envs/webchaser/bin/activate"
    "$HOME/miniconda3/envs/webdancer/bin/activate"
    "$HOME/anaconda3/envs/webdancer/bin/activate"
    "./venv/bin/activate"
)

VENV_ACTIVATED=false
for venv_path in "${VENV_PATHS[@]}"; do
    if [ -f "$venv_path" ]; then
        echo "Found virtual environment at: $venv_path"
        source "$venv_path"
        VENV_ACTIVATED=true
        break
    fi
done

if [ "$VENV_ACTIVATED" = false ]; then
    echo "Warning: No virtual environment found. Using system Python."
    echo "Searched paths:"
    for path in "${VENV_PATHS[@]}"; do
        echo "  - $path"
    done
fi

# 检查 gunicorn 是否可用
if ! command -v gunicorn &> /dev/null; then
    echo "Error: gunicorn not found. Please install it:"
    echo "  pip install gunicorn"
    exit 1
fi

# 检查 Python 路径和版本
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "Gunicorn path: $(which gunicorn)"

# 使用 gunicorn 启动服务
echo "Starting WebChasor API server..."
gunicorn -c src/api/gunicorn.conf.py src.api.api_server:app
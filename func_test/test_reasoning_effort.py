#!/usr/bin/env python3
"""
测试 OSS 20b 模型在不同推理等级（reasoning-effort）下的表现差异

使用场景：
- 测试同一个问题在不同推理等级下的回答质量
- 比较推理时间和回答长度
- 评估是否值得使用更高的推理等级

推理等级（OpenAI o1 系列参数）：
- low: 快速推理，适合简单问题
- medium: 中等推理，平衡速度和质量（默认）
- high: 深度推理，适合复杂问题
"""

import os
import time
from dotenv import load_dotenv
import openai

# 加载环境变量
load_dotenv()

# 从 config.yaml 读取配置
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from config_manager import get_config
cfg = get_config()

# 配置
OPENAI_API_BASE = cfg.get('external_services.openai.api_base', 'https://api.openai.com/v1')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_AGENT")
MODEL_NAME = cfg.get('models.planner.model_name', 'gpt-oss-120b')
# MODEL_NAME = 'HKGAI-Qwen3-32b'
# MODEL_NAME = 'web-dancer'
# 检查环境变量
if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY_AGENT not found in .env file")
    print("\nPlease set the following in your .env file:")
    print("  OPENAI_API_KEY_AGENT=your_api_key")
    exit(1)

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

print("=" * 80)
print("OSS 20b 模型推理等级测试")
print("=" * 80)
print(f"\n[CONFIG] API Base: {OPENAI_API_BASE}")
print(f"[CONFIG] Model: {MODEL_NAME}")
print(f"[CONFIG] API Key: {'✅ Set' if OPENAI_API_KEY else '❌ Not Set'}")
print("\n" + "=" * 80)


def test_reasoning_effort(query: str, effort_level: str = None):
    """
    测试指定推理等级下的模型表现
    
    Args:
        query: 测试问题
        effort_level: 推理等级 ('low', 'medium', 'high', 或 None 表示不指定)
    
    Returns:
        dict: 包含响应、耗时、tokens 等信息
    """
    print(f"\n{'▬' * 80}")
    print(f"[TEST] Reasoning Effort: {effort_level if effort_level else 'default (not specified)'}")
    print(f"{'▬' * 80}")
    
    # 构建请求参数
    messages = [
        {
            "role": "system", 
            "content": "你是一个精准的智能助手，能够深入分析问题并给出有见地的回答。"
        },
        {
            "role": "user", 
            "content": query
        }
    ]
    
    # API 调用参数
    api_params = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 10000
    }
    
    # 如果指定了推理等级，添加参数
    # 注意：reasoning_effort 是 OpenAI o1 系列的参数
    # 如果 OSS API 不支持，可能会被忽略或报错
    if effort_level:
        api_params["reasoning_effort"] = effort_level
    
    try:
        print(f"\n[REQUEST] Calling API with params:")
        print(f"  Model: {MODEL_NAME}")
        print(f"  Temperature: {api_params['temperature']}")
        print(f"  Max Tokens: {api_params['max_tokens']}")
        if effort_level:
            print(f"  Reasoning Effort: {effort_level}")
        
        time_start = time.time()
        response = client.chat.completions.create(**api_params)
        time_end = time.time()
        
        # 提取结果
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        usage = response.usage
        
        elapsed_time = time_end - time_start
        
        # 打印结果
        print(f"\n[RESPONSE] Status: Success")
        print(f"[RESPONSE] Time: {elapsed_time:.2f}s")
        print(f"[RESPONSE] Finish Reason: {finish_reason}")
        if usage:
            print(f"[RESPONSE] Tokens - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
        print(f"\n[CONTENT] Length: {len(content)} chars")
        print(f"[CONTENT] Response:\n")
        print("─" * 80)
        print(content)
        print("─" * 80)
        
        return {
            "effort_level": effort_level or "default",
            "content": content,
            "elapsed_time": elapsed_time,
            "finish_reason": finish_reason,
            "usage": usage,
            "char_count": len(content),
            "success": True
        }
        
    except Exception as e:
        print(f"\n[ERROR] API call failed: {e}")
        return {
            "effort_level": effort_level or "default",
            "error": str(e),
            "success": False
        }


def compare_results(results: list):
    """
    比较不同推理等级的结果
    
    Args:
        results: 测试结果列表
    """
    print("\n\n" + "█" * 80)
    print("结果对比")
    print("█" * 80)
    
    # 创建对比表格
    print(f"\n{'推理等级':<15} {'耗时(秒)':<12} {'字符数':<12} {'Tokens':<20} {'状态':<10}")
    print("─" * 80)
    
    for result in results:
        if result['success']:
            effort = result['effort_level']
            elapsed = f"{result['elapsed_time']:.2f}s"
            chars = result['char_count']
            tokens = f"{result['usage'].total_tokens}" if result['usage'] else "N/A"
            status = "✅ 成功"
        else:
            effort = result['effort_level']
            elapsed = "N/A"
            chars = "N/A"
            tokens = "N/A"
            status = "❌ 失败"
        
        print(f"{effort:<15} {elapsed:<12} {str(chars):<12} {tokens:<20} {status:<10}")
    
    print("\n" + "─" * 80)
    
    # 分析建议
    print("\n[ANALYSIS] 推理等级选择建议：")
    print("  - 如果 'high' 等级显著提升回答质量且时间可接受 → 使用 high")
    print("  - 如果 'medium' 和 'high' 差异不大 → 使用 medium（更快）")
    print("  - 如果 'low' 已足够且速度快很多 → 使用 low")
    print("  - 如果指定 reasoning_effort 报错 → OSS API 可能不支持此参数")
    

def main():
    """
    主测试流程
    """
    # 测试问题（从 planner.py 中的例子）
    test_query = "解读一下：八字：己巳 甲戌 壬戌 癸卯。 不要给出答案，只输出思考解决过程,尽你最大可能去猜测，给出一个最合理的方案快速准确解决问题。"
    
    print(f"\n[TEST QUERY] {test_query}\n")
    print("=" * 80)
    print("开始测试不同推理等级...")
    print("=" * 80)
    
    # 测试配置
    effort_levels = [
        None,      # 不指定（默认行为）
        "low",     # 低推理等级
        "medium",  # 中等推理等级
        "high"     # 高推理等级
    ]
    
    results = []
    
    # 逐个测试
    for effort in effort_levels:
        result = test_reasoning_effort(test_query, effort)
        results.append(result)
        
        # 等待一下，避免 API rate limit
        if effort != effort_levels[-1]:
            print("\n⏳ Waiting 2 seconds before next test...")
            time.sleep(2)
    
    # 对比结果
    compare_results(results)
    
    print("\n\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()


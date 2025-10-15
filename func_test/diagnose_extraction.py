#!/usr/bin/env python3
"""
诊断提取流程：检查 Web Scraping → Ranking → Extraction 各阶段
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from config_manager import get_config

# 检查配置
cfg = get_config()

print("="*80)
print("🔍 提取流程诊断")
print("="*80)

print("\n1️⃣ Web Scraping 配置检查")
print("-"*80)
web_scraping_enabled = cfg.get('ir_rag.web_scraping.enabled', False)
max_pages = cfg.get('ir_rag.web_scraping.max_pages', 0)
print(f"enabled: {web_scraping_enabled}")
print(f"max_pages: {max_pages}")

if not web_scraping_enabled:
    print("❌ 问题：Web scraping 未启用！这就是为什么只有 SERP_SNIPPET")
    print("   解决：在 config.yaml 中设置 ir_rag.web_scraping.enabled: true")
else:
    print("✅ Web scraping 已启用")

print("\n2️⃣ Extraction 配置检查")
print("-"*80)
snippet_threshold = cfg.get('ir_rag.extraction.snippet_confidence_threshold', 0.75)
max_passages = cfg.get('ir_rag.content.max_passages_per_task', 3)
print(f"snippet_confidence_threshold: {snippet_threshold}")
print(f"max_passages_per_task: {max_passages}")

if snippet_threshold > 0.7:
    print("⚠️  提示：阈值较高 (>0.7)，可能导致很多任务进入 Stage 2")
    print("   建议：降低到 0.65 可以让更多任务在 Stage 1 就满意")

if max_passages < 5:
    print("⚠️  提示：max_passages 较少，可能错过关键信息")
    print("   建议：增加到 6 可以提供更多上下文")

print("\n3️⃣ Search 配置检查")
print("-"*80)
max_results = cfg.get('ir_rag.search.max_results', 10)
providers = cfg.get('ir_rag.search.provider', 'serpapi')
print(f"max_results: {max_results}")
print(f"provider: {providers}")

print("\n4️⃣ Ranker 配置检查")
print("-"*80)
ranking_algo = cfg.get('ir_rag.ranking.algorithm', 'keyword_matching')
entity_weight = cfg.get('ir_rag.ranking.entity_weight', 3.5)
keyword_weight = cfg.get('ir_rag.ranking.keyword_weight', 1.0)
print(f"algorithm: {ranking_algo}")
print(f"entity_weight: {entity_weight}")
print(f"keyword_weight: {keyword_weight}")

print("\n" + "="*80)
print("📋 建议修改")
print("="*80)

recommendations = []

if not web_scraping_enabled:
    recommendations.append({
        "priority": "🔴 P0 - 最高优先级",
        "issue": "Web scraping 未启用",
        "fix": "config.yaml 中设置:\nir_rag:\n  web_scraping:\n    enabled: true\n    max_pages: 5"
    })

if snippet_threshold > 0.7:
    recommendations.append({
        "priority": "🟡 P1 - 建议优化",
        "issue": f"snippet_confidence_threshold 过高 ({snippet_threshold})",
        "fix": "config.yaml 中设置:\nir_rag:\n  extraction:\n    snippet_confidence_threshold: 0.65"
    })

if max_passages < 5:
    recommendations.append({
        "priority": "🟡 P1 - 建议优化",
        "issue": f"max_passages_per_task 较少 ({max_passages})",
        "fix": "config.yaml 中设置:\nir_rag:\n  content:\n    max_passages_per_task: 6"
    })

if recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['priority']}")
        print(f"   问题：{rec['issue']}")
        print(f"   修复：{rec['fix']}")
else:
    print("\n✅ 配置看起来没有明显问题")
    print("   如果提取仍然失败，可能是：")
    print("   1. Wikipedia 页面没有被选中（URL 选择逻辑问题）")
    print("   2. Ranker 给 Wikipedia 内容的分数太低")
    print("   3. Wikipedia 内容中确实没有需要的信息")

print("\n" + "="*80)


#!/usr/bin/env python3
"""
测试 Planner 的完整性原则（Completeness Principle）
验证是否为组织/实体查询生成 WHO 维度任务
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv()

import openai
from planner import Planner
from config_manager import get_config

# ============================================================================
# 测试配置
# ============================================================================

# 测试查询：涵盖不同类型的组织/实体
TEST_QUERIES = [
    # 政府组织
    {
        "query": "香港三司十五局",
        "expected_who_tasks": ["司长", "局长", "首长", "负责人", "leaders", "heads", "chiefs"],
        "category": "政府"
    },
    # 公司组织
    {
        "query": "特斯拉公司组织架构",
        "expected_who_tasks": ["CEO", "高管", "负责人", "executives", "leadership", "heads"],
        "category": "公司"
    },
    # 大学组织
    {
        "query": "清华大学院系设置",
        "expected_who_tasks": ["校长", "院长", "系主任", "president", "dean", "chair"],
        "category": "大学"
    },
    # 球队组织
    {
        "query": "巴塞罗那足球队阵容",
        "expected_who_tasks": ["教练", "队长", "球员", "coach", "captain", "player"],
        "category": "球队"
    },
    # 国际组织
    {
        "query": "联合国安理会组成",
        "expected_who_tasks": ["秘书长", "代表", "主席", "secretary", "representative", "chair"],
        "category": "国际组织"
    }
]

# ============================================================================
# 测试函数
# ============================================================================

def check_has_who_dimension(plan, expected_keywords):
    """
    检查计划中是否包含 WHO 维度的任务
    
    Returns:
        (has_who, matched_tasks, score)
    """
    matched_tasks = []
    
    for task in plan.tasks_to_extract:
        task_text = (task.fact + task.variable_name).lower()
        
        # 检查是否匹配任何预期的关键词
        for keyword in expected_keywords:
            if keyword.lower() in task_text:
                matched_tasks.append({
                    'variable_name': task.variable_name,
                    'fact': task.fact,
                    'matched_keyword': keyword
                })
                break
    
    has_who = len(matched_tasks) > 0
    score = min(100, len(matched_tasks) * 50)  # 每个匹配任务 50 分，最高 100
    
    return has_who, matched_tasks, score


async def test_single_query(planner, test_case):
    """测试单个查询"""
    query = test_case["query"]
    expected_who = test_case["expected_who_tasks"]
    category = test_case["category"]
    
    print(f"\n{'='*80}")
    print(f"测试查询：{query} [{category}]")
    print(f"{'='*80}")
    
    # 生成计划
    plan = await planner.plan(query)
    
    # 显示生成的任务
    print(f"\n生成的任务数量：{len(plan.tasks_to_extract)}")
    print(f"\n任务列表：")
    for i, task in enumerate(plan.tasks_to_extract, 1):
        print(f"  {i}. [{task.category}] {task.variable_name}")
        print(f"     Fact: {task.fact}")
    
    # 检查是否包含 WHO 维度
    has_who, matched_tasks, score = check_has_who_dimension(plan, expected_who)
    
    print(f"\n{'─'*80}")
    print(f"WHO 维度检查：")
    if has_who:
        print(f"✅ 通过 - 找到 {len(matched_tasks)} 个相关任务")
        for match in matched_tasks:
            print(f"   - {match['variable_name']} (匹配: {match['matched_keyword']})")
            print(f"     {match['fact']}")
    else:
        print(f"❌ 失败 - 未找到 WHO 维度任务")
        print(f"   预期关键词: {', '.join(expected_who[:5])}...")
    
    print(f"\n评分: {score}/100")
    
    return {
        "query": query,
        "category": category,
        "total_tasks": len(plan.tasks_to_extract),
        "has_who": has_who,
        "who_tasks_count": len(matched_tasks),
        "score": score,
        "passed": has_who
    }


async def main():
    """主测试流程"""
    print("="*80)
    print("Planner 完整性原则测试")
    print("="*80)
    print("\n测试目标：验证 Planner 是否为组织/实体查询生成 WHO 维度任务")
    
    # 初始化配置
    cfg = get_config()
    api_base = cfg.get('external_services.openai.api_base')
    api_key = os.getenv("OPENAI_API_KEY_AGENT")
    model_name = cfg.get('models.planner.model_name', 'gpt-oss-20b')
    
    if not api_key:
        print("\n❌ 错误: 未找到 OPENAI_API_KEY_AGENT 环境变量")
        return 1
    
    print(f"\n配置:")
    print(f"  API Base: {api_base}")
    print(f"  Model: {model_name}")
    print(f"  测试数量: {len(TEST_QUERIES)}")
    
    # 初始化 Planner
    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    planner = Planner(llm_client=client, model_name=model_name)
    
    # 运行测试
    results = []
    for test_case in TEST_QUERIES:
        result = await test_single_query(planner, test_case)
        results.append(result)
        
        # 等待避免 rate limit
        await asyncio.sleep(1)
    
    # 汇总结果
    print(f"\n\n{'█'*80}")
    print("测试汇总")
    print(f"{'█'*80}")
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    avg_score = sum(r['score'] for r in results) / total if total > 0 else 0
    
    print(f"\n总体结果：")
    print(f"  通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"  平均分: {avg_score:.1f}/100")
    
    print(f"\n详细结果：")
    print(f"{'类别':<12} {'查询':<25} {'任务数':<8} {'WHO任务':<10} {'评分':<8} {'状态'}")
    print("─"*80)
    
    for r in results:
        status = "✅ 通过" if r['passed'] else "❌ 失败"
        print(f"{r['category']:<12} {r['query']:<25} {r['total_tasks']:<8} "
              f"{r['who_tasks_count']:<10} {r['score']:<8} {status}")
    
    # 评估
    print(f"\n{'─'*80}")
    print("评估：")
    if passed / total >= 0.8:
        print("✅ 优秀 - 步骤1（Prompt修改）效果显著，建议直接使用")
    elif passed / total >= 0.6:
        print("⚠️  良好 - 步骤1有效，但建议考虑添加步骤3（代码检查）提升稳定性")
    else:
        print("❌ 不足 - 需要添加步骤3（代码层检查）来保证准确性")
    
    print("="*80)
    return 0 if passed / total >= 0.6 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


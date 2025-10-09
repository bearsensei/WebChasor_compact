"""
QueryMaker Test Script
Test query generation for different types of queries
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[TEST][QueryMaker] Loaded environment from {env_path}\n")
    else:
        print(f"[TEST][QueryMaker] .env file not found at {env_path}\n")
except ImportError:
    print("[TEST][QueryMaker] python-dotenv not installed, skipping .env file loading\n")

from actions.querymaker import QueryMaker
from artifacts import Context
from utils.timectx import TimeContext

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# Test queries covering different types
TEST_QUERIES = [
    # 1. Simple queries (lazy mode - just a name or noun)
    "郭毅可",           # Just a person name
    "稳定币",           # Just a term
    "NVDA",            # Stock ticker
    "浸会大学",         # Institution name
    
    # 2. Definition queries (explicit questions)
    "什么是稳定币？",
    "What is a stablecoin?",
    
    # 3. Complex factual query (needs diverse queries)
    "比较香港和新加坡加密货币的优势和劣势",
    
    # 4. Data/comparison query (needs multiple angles)
    "香港各大学校长年薪对比",
    
    # 5. How-to query
    "如何提升大模型推理能力？",
]

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def print_section(title: str):
    """Print a section divider"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def analyze_query_type(query: str) -> dict:
    """
    Analyze query type to determine optimal number of queries
    """
    query_lower = query.lower()
    query_stripped = query.strip()
    
    # Simple query detection (懒人模式：只输入一个名词/名字)
    is_simple = (
        len(query_stripped) <= 20 and  # 短查询
        '?' not in query_stripped and '？' not in query_stripped and  # 没有问号
        not any(keyword in query_lower for keyword in [
            '什么', '如何', '怎么', '为什么', '哪', '多少',
            'what', 'how', 'why', 'when', 'where', 'which', 'who'
        ]) and  # 没有疑问词
        len(query_stripped.split()) <= 4  # 词数很少
    )
    
    # Definition queries (什么是, 是什么, what is, what are)
    is_definition = any(keyword in query_lower for keyword in [
        '什么是', '是什么', 'what is', 'what are', '定义', 'definition'
    ])
    
    # Comparison queries (对比, 比较, compare, vs, versus)
    is_comparison = any(keyword in query_lower for keyword in [
        '对比', '比较', 'compare', 'comparison', 'vs', 'versus', '差异'
    ])
    
    # How-to queries (如何, 怎么, how to, how can)
    is_howto = any(keyword in query_lower for keyword in [
        '如何', '怎么', '怎样', 'how to', 'how can', 'how do'
    ])
    
    # Time-sensitive queries (什么时候, when, 现在, latest)
    is_time_sensitive = any(keyword in query_lower for keyword in [
        '什么时候', '何时', 'when', '现在', '最新', 'latest', 'current'
    ])
    
    # Complex factual queries (who, where, why with multiple entities)
    is_complex_factual = any(keyword in query_lower for keyword in [
        '谁', 'who', '为什么', 'why', '原因', 'reason'
    ]) or query.count('的') >= 2  # Multiple entities in Chinese
    
    return {
        'is_simple': is_simple,
        'is_definition': is_definition,
        'is_comparison': is_comparison,
        'is_howto': is_howto,
        'is_time_sensitive': is_time_sensitive,
        'is_complex_factual': is_complex_factual,
        'recommended_queries': get_recommended_query_count(
            is_simple, is_definition, is_comparison, is_howto, is_time_sensitive, is_complex_factual
        )
    }

def get_recommended_query_count(is_simple, is_def, is_comp, is_howto, is_time, is_complex) -> int:
    """Determine optimal query count based on query type (priority order matters!)"""
    # Comparison and How-to should have higher priority than Simple
    if is_comp:
        return 8  # Comparison: 需要多角度 (highest priority)
    elif is_howto:
        return 5  # How-to: 需要方法+步骤+案例
    elif is_def:
        return 2  # Definition: 原始查询 + 1个变体
    elif is_simple:
        return 2  # Simple: 原始查询 + 1个扩展（加"介绍"）
    elif is_time and is_complex:
        return 10  # Complex time-sensitive: 需要全面覆盖
    elif is_complex:
        return 8  # Complex factual: 需要多来源验证
    else:
        return 5  # Default

async def test_single_query(querymaker: QueryMaker, query: str):
    """Test query generation for a single query"""
    print_section(f"Testing Query: {query}")
    
    # Analyze query type
    analysis = analyze_query_type(query)
    print(f"\n[ANALYSIS] Query Type:")
    print(f"  - Simple (lazy mode): {analysis['is_simple']}")
    print(f"  - Definition: {analysis['is_definition']}")
    print(f"  - Comparison: {analysis['is_comparison']}")
    print(f"  - How-to: {analysis['is_howto']}")
    print(f"  - Time-sensitive: {analysis['is_time_sensitive']}")
    print(f"  - Complex factual: {analysis['is_complex_factual']}")
    print(f"  - Recommended query count: {analysis['recommended_queries']}")
    
    if analysis['is_simple']:
        print(f"\n💡 [LAZY MODE] This is a simple query - will generate minimal queries")
    
    # Create context with required fields
    ctx = Context(
        query=query,
        history="",
        router_category="IR_RAG"
    )
    
    # Generate queries
    print(f"\n[GENERATING] Calling QueryMaker...")
    queries = querymaker.generate_queries(query, ctx)
    
    # Display results
    print(f"\n[RESULT] Generated {len(queries)} queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    
    # Analysis
    if len(queries) > analysis['recommended_queries']:
        print(f"\n[RECOMMENDATION] For this type of query, {analysis['recommended_queries']} queries would be sufficient.")
        print(f"                  Consider reducing from {len(queries)} to {analysis['recommended_queries']}.")
    elif len(queries) < analysis['recommended_queries']:
        print(f"\n[RECOMMENDATION] For this type of query, {analysis['recommended_queries']} queries recommended.")
        print(f"                  Consider increasing from {len(queries)} to {analysis['recommended_queries']}.")
    else:
        print(f"\n[RECOMMENDATION] Query count ({len(queries)}) is optimal for this type.")
    
    return queries, analysis

async def test_all_queries():
    """Test all predefined queries"""
    print_section("QueryMaker Test Suite")
    
    # Initialize QueryMaker
    print("\n[INIT] Initializing QueryMaker...")
    querymaker = QueryMaker()
    
    results = []
    
    for query in TEST_QUERIES:
        queries, analysis = await test_single_query(querymaker, query)
        results.append({
            'original': query,
            'generated': queries,
            'count': len(queries),
            'recommended': analysis['recommended_queries'],
            'analysis': analysis
        })
        
        # Wait between requests to avoid rate limiting
        await asyncio.sleep(1)
    
    # Summary
    print_section("Test Summary")
    
    print("\n[SUMMARY] Query Generation Results:")
    print(f"\n{'Original Query':<40} {'Generated':<12} {'Recommended':<12} {'Status'}")
    print("-" * 80)
    
    for r in results:
        status = "✓ Optimal" if r['count'] == r['recommended'] else "⚠ Suboptimal"
        print(f"{r['original']:<40} {r['count']:<12} {r['recommended']:<12} {status}")
    
    # Recommendations
    print("\n[RECOMMENDATIONS]")
    
    simple_queries = [r for r in results if r['analysis']['is_simple']]
    if simple_queries:
        avg_generated = sum(r['count'] for r in simple_queries) / len(simple_queries)
        avg_recommended = sum(r['recommended'] for r in simple_queries) / len(simple_queries)
        print(f"\n1. Simple queries (lazy mode):")
        print(f"   - Count: {len(simple_queries)}")
        print(f"   - Average generated: {avg_generated:.1f}")
        print(f"   - Recommended: {avg_recommended:.1f}")
        if avg_generated <= avg_recommended:
            print(f"   ✓ Optimized for lazy users!")
    
    definition_queries = [r for r in results if r['analysis']['is_definition']]
    if definition_queries:
        avg_generated = sum(r['count'] for r in definition_queries) / len(definition_queries)
        avg_recommended = sum(r['recommended'] for r in definition_queries) / len(definition_queries)
        print(f"\n2. Definition queries:")
        print(f"   - Count: {len(definition_queries)}")
        print(f"   - Average generated: {avg_generated:.1f}")
        print(f"   - Recommended: {avg_recommended:.1f}")
        if avg_generated > avg_recommended:
            print(f"   - Action: Reduce query count for definition-type queries")
    
    comparison_queries = [r for r in results if r['analysis']['is_comparison']]
    if comparison_queries:
        print(f"\n3. Comparison queries:")
        print(f"   - Count: {len(comparison_queries)}")
        print(f"   - These require more diverse queries for comprehensive coverage")
    
    print("\n[DONE] Test completed successfully")

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main test function"""
    try:
        await test_all_queries()
        return 0
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


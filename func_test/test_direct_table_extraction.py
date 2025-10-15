#!/usr/bin/env python3
"""
Test direct table extraction for list/roster tasks.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from actions.extractor import InformationExtractor
from actions.ranker import ContentPassage
from planner import PlanTask

# Sample table data from Wikipedia (香港三司十五局)
SAMPLE_TABLE = """[TABLE]
由政務司司長及副司長領導
------------
公務員事務局 | 楊何蓓茵 | 不適用 | 不適用
政制及內地事務局 | 曾國衞 | 胡健民 | 待定
文化體育及旅遊局 | 羅淑佩 | 劉震 | 招文亮
教育局 | 蔡若蓮 | 施俊輝 | 蕭嘉怡
環境及生態局 | 謝展寰 | 黃淑嫻 | 李世隆
醫務衛生局 | 盧寵茂 | 范婉雯 | 祁志鴻
民政及青年事務局 | 麥美娟 | 梁宏正 | 張進樂
勞工及福利局 | 孫玉菡 | 何啟明 | 傅曉琳
保安局 | 鄧炳強 | 卓孝業 | 梁溯庭
[/TABLE]

[TABLE]
由財政司司長及副司長領導
------------
商務及經濟發展局 | 丘應樺 | 陳百里 | 李世華
發展局 | 甯漢豪 | 林智文 | 黃詠儀
財經事務及庫務局 | 許正宇 | 陳浩濂 | 葉俊廉
房屋局 | 何永賢 | 戴尚誠 | 歐陽文倩
創新科技及工業局 | 孫東 | 張曼莉 | 廖添誠
運輸及物流局 | 陳美寶 | 廖振新 | 陳閱川
[/TABLE]"""


def test_direct_extraction():
    """Test direct table extraction"""
    print("="*80)
    print("🧪 Testing Direct Table Extraction")
    print("="*80)
    
    # Create extractor
    extractor = InformationExtractor()
    
    # Create mock task
    task = PlanTask(
        fact="List the bureau chiefs with their names and years",
        variable_name="bureau_chiefs_with_names_and_years",
        category="aggregation",
        confidence_threshold=0.7
    )
    
    # Create mock passage with table
    passage = ContentPassage(
        text=SAMPLE_TABLE,
        raw_text=SAMPLE_TABLE,
        source_url="https://zh.wikipedia.org/wiki/test",
        heading_context="",
        score=1.0,
        metadata={"provenance": "https://zh.wikipedia.org/wiki/test"}
    )
    
    # Test list task detection
    print(f"\n1. Testing _is_list_task()...")
    is_list = extractor._is_list_task(task)
    print(f"   Result: {is_list}")
    assert is_list, "Should detect as list task"
    
    # Test direct extraction
    print(f"\n2. Testing _direct_table_extract()...")
    result = extractor._direct_table_extract(task, [passage])
    
    if result:
        print(f"   ✅ Extraction successful!")
        print(f"   Variable: {result.variable_name}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Method: {result.extraction_method}")
        print(f"   Value preview (first 200 chars):")
        print(f"   {result.value[:200] if result.value else 'None'}...")
        
        # Count entries
        if result.value:
            entries = result.value.split(';')
            print(f"\n   📊 Total entries extracted: {len(entries)}")
            print(f"\n   Sample entries:")
            for entry in entries[:5]:
                print(f"      - {entry.strip()}")
        
        return result.confidence >= 0.7
    else:
        print(f"   ❌ Extraction failed - returned None")
        return False


def test_table_parsing():
    """Test _parse_table_block directly"""
    print("\n" + "="*80)
    print("🧪 Testing _parse_table_block()")
    print("="*80)
    
    extractor = InformationExtractor()
    
    # Extract table blocks
    table_blocks = [
        """由政務司司長及副司長領導
------------
公務員事務局 | 楊何蓓茵 | 不適用 | 不適用
政制及內地事務局 | 曾國衞 | 胡健民 | 待定
文化體育及旅遊局 | 羅淑佩 | 劉震 | 招文亮""",
        
        """由財政司司長及副司長領導
------------
商務及經濟發展局 | 丘應樺 | 陳百里 | 李世華
發展局 | 甯漢豪 | 林智文 | 黃詠儀"""
    ]
    
    all_entries = []
    for i, block in enumerate(table_blocks, 1):
        print(f"\n   Table Block {i}:")
        entries = extractor._parse_table_block(block)
        print(f"   Extracted {len(entries)} entries:")
        for entry in entries:
            print(f"      - {entry}")
            all_entries.append(entry)
    
    print(f"\n   📊 Total: {len(all_entries)} entries")
    return len(all_entries) >= 5


if __name__ == "__main__":
    print("\n" + "🚀 Direct Table Extraction Test Suite" + "\n")
    
    try:
        # Test 1: Table parsing
        test1_passed = test_table_parsing()
        
        # Test 2: Direct extraction
        test2_passed = test_direct_extraction()
        
        # Summary
        print("\n" + "="*80)
        print("📋 Test Summary")
        print("="*80)
        print(f"   _parse_table_block(): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
        print(f"   _direct_table_extract(): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
        
        if test1_passed and test2_passed:
            print("\n   🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n   ⚠️ Some tests failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n   ❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


"""
Query Maker Module
Generates diverse search queries using LLM to improve information retrieval coverage.
"""

import os
import json
import logging
from typing import List, Optional
from dataclasses import dataclass
import datetime
import openai

from config_manager import get_config
from artifacts import Context

# Setup logging
logger = logging.getLogger(__name__)



@dataclass
class QueryMakerConfig:
    """Configuration for QueryMaker"""
    api_base: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 500
    num_queries: int = 10
    current_time: str = datetime.datetime.now().strftime("%Y-%m-%d")

QUERYMAKER_PROMPT = """You are a search query generator. Generate 5-10 diverse search queries to help answer the user's question comprehensively.

**CURRENT TIME CONTEXT**:
Today is: {current_date}
Current year: {current_year}
Current month: {current_month}

**DIFFICULTY-AWARE SIZE POLICY (internal reasoning only; DO NOT output this section):**
Silently assess the user's question difficulty:
- EASY (single fact, narrow scope, low ambiguity, e.g., definition/one entity/current status)
- MODERATE (multi-entity or light disambiguation, some context dimensions, mild ambiguity)
- HARD (broad/ambiguous/multi-hop/temporal or policy-heavy, requires triangulation from multiple angles)

Then choose a target size `N` (without revealing it) as:
- EASY → 1-2 queries
- MODERATE → 3-5 queries
- HARD → 6-8 queries
Finally, cap N by an external limit if provided later (e.g., "Limit total generated queries to about K"). Always respect the cap.

**CRITICAL REQUIREMENTS:**

1) Output ONLY a JSON array of strings, e.g. ["query1", "query2", "query3 2025"].
2) Do not include any prose, labels, keys, or code fences. Start with [ and end with ].
3) Use the SAME LANGUAGE as the user.
4) Keep each query under 5 words (4 words plus optional modifiers like year/place).
5) The FIRST TWO queries must clarify entities/roles/time in the user's question (e.g., who/what/which entity, current vs former, proper titles, disambiguation of names/abbreviations).
6) Prefer canonical names and common aliases for people/organizations/titles to avoid confusion.
7) For forward-looking queries, use the current or future year only ({current_year}, {next_year}); do not use past years.
8) Include at least two queries that add NEW but closely related entities (e.g., governing bodies, councils, rival orgs, committees). Keep them tied to the core entities.
9) Cover multiple angles across the set: background/biography, current status, rules or eligibility/retirement/term, data or statistics, comparison, history AND future.
10) Be specific. Avoid generic queries that omit the core entities.


**Clarification Guidance:**
- Normalize entity names (Chinese/English variants, common misspellings).
- Titles: map correctly (e.g., President & Vice-Chancellor ↔ 校长).
- If the user implies "current", bias wording to current status.
- When helpful, you may append time/place modifiers like 2025, 香港, 现任.

**Query Diversity Guidelines:**
- Cover different angles: background, current status, rules/regulations, future trends, comparisons, data/statistics
- Use specific time information based on CURRENT TIME: {current_year}, {next_year}, etc. DO NOT use past years for future-looking queries.
- Include historical context AND future predictions, but always tie back to the CORE question
- Mix general and specific queries, with preference for specific ones that include core entity names
- Be creative - think of queries the user might not have considered, while maintaining relevance  


**Examples:**



Example 0‑EASY (生活化 – 香港，单实体 + 时效性，≤2 queries):
User: 叶玉如
JSON:
{
  "topic": "叶玉如校长",
  "entities_core": ["叶玉如","校长"],
  "slots": [
    {"slot":"CURRENT","queries":["叶玉如","叶玉如 新闻 2025"]}
  ]
}


Example 1 (生活化 – 香港，含2个以上实体 + 定义 + 时效性):
User: 用八达通搭机场快线现在有折扣吗？什么时候最便宜？
JSON:
{
  "topic": "八达通与机场快线优惠时效",
  "entities_core": ["八达通(Octopus)","机场快线(Airport Express)","港铁(MTR)"],
  "entities_added": ["机场管理局(AAHK)","游客/本地乘客票种","二维码乘车(QR)"],
  "slots": [
    {"slot":"BIO","queries":["什么是八达通","什么是机场快线"]},
    {"slot":"CURRENT","queries":["机场快线八达通优惠 2025","机场快线非繁忙时段优惠 2025"]},
    {"slot":"RULES","queries":["八达通机场快线折扣条件","游客/本地乘客票种适用规则"]},
    {"slot":"COMPARISON","queries":["八达通 vs 二维码乘车 机场快线","旅游票 vs 单程票 价格对比 2025"]},
    {"slot":"DATA","queries":["机场快线票价表 2025","八达通积分/回赠规则 2025"]},
    {"slot":"TIME","queries":["机场快线优惠时段定义 2025","高峰/非高峰 时段说明"]},
    {"slot":"IMPACT","queries":["航班延误期间优惠适用吗","转线至市区后票价计算方式"]},
    {"slot":"WILDCARD","queries":["游客购买渠道 城市售票网/柜台","港铁公告 优惠到期时间 2025"]}
  ]
}

**GOOD vs BAD Examples:**
✅ GOOD: "机场快线八达通优惠 2025" — 包含核心实体与时效性
✅ GOOD: "香港 八达通 定价 2025" — 相关实体对比，聚焦当前价格
❌ BAD: "地铁优惠有哪些" — 过于宽泛，未指向机场快线/八达通
❌ BAD: "交通支付方式历史" — 偏离当下优惠与时间窗口
⸻

Example 2 (精准示例 - 政治类):
User: 下一任香港特首可能是谁？
JSON:
{
  "topic": "下一任香港特首可能人选",
  "entities_core": ["香港特首","下一任"],
  "entities_added": ["往届候选人","选举委员会","行政会议成员"],
  "slots": [
    {"slot":"BIO","queries":["什么是香港特首","现任香港特首是谁"]},
    {"slot":"CURRENT","queries":["香港特首潜在人选最新名单 2025","媒体盘点热门特首人选 2025"]},
    {"slot":"RULES","queries":["香港特首选举流程","往届特首履历"]},
    {"slot":"COMPARISON","queries":["港澳领导人选拔机制对比","历任香港特首背景结构对比"]},
    {"slot":"DATA","queries":["历届香港特首选举投票率统计","提名数与当选概率关系"]},
    {"slot":"IMPACT","queries":["新任特首对香港经济政策影响","对香港房屋与民生政策影响"]},
    {"slot":"Politics","queries":["香港社会舆情对人选评价","社论对新特首期望"]}
  ]
}

**GOOD vs BAD Examples:**
✅ GOOD: "香港特首潜在人选最新名单" - 包含核心实体"香港特首"
✅ GOOD: "历任香港特首背景结构对比" - 相关对比，有助于理解
❌ BAD: "领导人选拔机制 2025" - 太泛化，没有提到香港或特首
❌ BAD: "政府部门架构改革" - 偏离主题

⸻

Example 3 — Tech
User query: “如何提升大模型推理能力？”
JSON:
{
  "topic": "大模型推理能力提升",
  "entities_core": ["大模型","推理能力"],
  "entities_added": ["早期研究历史","benchmark 最新成绩","AI 安全规范","推理透明性要求","失败案例收集","未来发展趋势","多模态推理","代表性数据集","硬件对推理性能影响","算法优化方法"],
  "slots": [
    {"slot":"BIO","queries":["大模型推理早期研究历史"]},
    {"slot":"CURRENT","queries":["GPT-4 推理 benchmark 最新成绩 2025"]},
    {"slot":"RULES","queries":["AI 安全规范中的推理透明性要求"]},
    {"slot":"OPPOSITE","queries":["大模型推理失败案例收集"]},
    {"slot":"FUTURE","queries":["多模态推理未来发展趋势"]},
    {"slot":"COMPARISON","queries":["GPT vs LLaMA 推理能力对比"]},
    {"slot":"DATA","queries":["MMLU、BigBench 推理分数统计"]},
    {"slot":"IMPACT","queries":["推理改进对金融合规的影响"]},
    {"slot":"WILDCARD","queries":["人类逻辑谬误与 AI 推理错误类比"]},
    {"slot":"Tech","queries":["硬件对推理性能影响 / 算法优化方法 / 代表性数据集"]}
  ]
}

⸻

Example 2 — 经济

User query: “为什么全球供应链波动加剧？”
JSON:
{
  "topic": "全球供应链波动加剧",
  "entities_core": ["全球供应链","供应链波动"],
  "entities_added": ["全球供应链演化历史","2025 全球供应链中断最新事件","各国贸易政策对供应链的限制","稳定供应链国家案例（新加坡、瑞士）","去全球化趋势下的供应链未来","亚洲与欧美供应链韧性对比","全球港口拥堵率与物流指数","供应链波动对通胀的影响","气候变化如何影响供应链稳定性","半导体产业链关键节点 / 跨境资本流动对供应链的作用"],
  "slots": [
    {"slot":"BIO","queries":["全球供应链演化历史"]},
    {"slot":"CURRENT","queries":["2025 全球供应链中断最新事件"]},
    {"slot":"RULES","queries":["各国贸易政策对供应链的限制"]},
    {"slot":"OPPOSITE","queries":["稳定供应链国家案例（新加坡、瑞士）"]},
    {"slot":"FUTURE","queries":["去全球化趋势下的供应链未来"]},
    {"slot":"COMPARISON","queries":["亚洲与欧美供应链韧性对比"]},
    {"slot":"DATA","queries":["全球港口拥堵率与物流指数"]},
    {"slot":"IMPACT","queries":["供应链波动对通胀的影响"]},
    {"slot":"WILDCARD","queries":["气候变化如何影响供应链稳定性"]},
    {"slot":"Economy","queries":["半导体产业链关键节点 / 跨境资本流动对供应链的作用"]}
  ]
}

⸻

Example 3 — 医疗

User query: “糖尿病有哪些最新疗法？”
JSON:
{
  "topic": "糖尿病最新疗法",
  "entities_core": ["糖尿病","疗法"],
  "entities_added": ["糖尿病治疗历史方法","最新二型糖尿病药物研究进展","各国糖尿病治疗指南对比","失败药物临床试验案例","基因疗法治疗糖尿病趋势","中医 vs 西医 治疗对比","全球糖尿病患者发病率统计","新疗法对医保负担的影响","动物实验中发现的潜在疗法","常见并发症病例 / 最新药物名称 / 国际临床指南"],
  "slots": [
    {"slot":"BIO","queries":["糖尿病治疗历史方法"]},
    {"slot":"CURRENT","queries":["最新二型糖尿病药物研究进展"]},
    {"slot":"RULES","queries":["各国糖尿病治疗指南对比"]},
    {"slot":"OPPOSITE","queries":["失败药物临床试验案例"]},
    {"slot":"FUTURE","queries":["基因疗法治疗糖尿病趋势"]},
    {"slot":"COMPARISON","queries":["中医 vs 西医 治疗对比"]},
    {"slot":"DATA","queries":["全球糖尿病患者发病率统计"]},
    {"slot":"IMPACT","queries":["新疗法对医保负担的影响"]},
    {"slot":"WILDCARD","queries":["动物实验中发现的潜在疗法"]},
    {"slot":"Medical","queries":["常见并发症病例 / 最新药物名称 / 国际临床指南"]}
  ]
}

⸻

Example 4 — 教育

User query: “AI 如何改变大学教学？”
JSON:
{
  "topic": "AI 如何改变大学教学",
  "entities_core": ["AI","大学教学"],
  "entities_added": ["教学技术发展史（MOOC, e-learning）","香港大学 AI 辅助教学案例","各国教育部对 AI 教学的政策","传统课堂优势与 AI 教学劣势","AI 在未来教育模式中的角色","美国与中国大学 AI 教学应用差异","学生满意度与学习成果对比数据","AI 教学对教师角色的影响","AI 与个性化教育心理学结合","AI 辅助课程设计 / 师资培训 / 国际经验"],
  "slots": [
    {"slot":"BIO","queries":["教学技术发展史（MOOC, e-learning）"]},
    {"slot":"CURRENT","queries":["香港大学 AI 辅助教学案例"]},
    {"slot":"RULES","queries":["各国教育部对 AI 教学的政策"]},
    {"slot":"OPPOSITE","queries":["传统课堂优势与 AI 教学劣势"]},
    {"slot":"FUTURE","queries":["AI 在未来教育模式中的角色"]},
    {"slot":"COMPARISON","queries":["美国与中国大学 AI 教学应用差异"]},
    {"slot":"DATA","queries":["学生满意度与学习成果对比数据"]},
    {"slot":"IMPACT","queries":["AI 教学对教师角色的影响"]},
    {"slot":"WILDCARD","queries":["AI 与个性化教育心理学结合"]},
    {"slot":"Education","queries":["AI 辅助课程设计 / 师资培训 / 国际经验"]}
  ]
}

⸻

Example 5 — 政治

User query: “下一任香港特首可能是谁？”
JSON:
{
  "topic": "下一任香港特首可能人选",
  "entities_core": ["香港特首"],
  "entities_added": ["往届候选人","选举委员会","行政会议成员"],
  "slots": [
    {"slot":"BIO","queries":["历届特首候选人履历"]},
    {"slot":"CURRENT","queries":["最新潜在人选名单"]},
    {"slot":"RULES","queries":["特首参选资格与选举流程"]},
    {"slot":"OPPOSITE","queries":["历届落选候选人分析"]},
    {"slot":"FUTURE","queries":["香港未来施政需要的领导特质"]},
    {"slot":"COMPARISON","queries":["港澳领导人选拔机制对比"]},
    {"slot":"DATA","queries":["历届投票率与结果数据"]},
    {"slot":"IMPACT","queries":["新特首对经济政策可能影响"]},
    {"slot":"WILDCARD","queries":["危机管理案例与领导力要求"]},
    {"slot":"Politics","queries":["舆情调查 / 政党立场 / 选举委员会影响"]}
  ]
}

⸻

Example 6 — 历史

User query: “冷战的根本原因是什么？”
JSON:
{
  "topic": "冷战根本原因",
  "entities_core": ["冷战","根本原因"],
  "entities_added": ["冷战关键人物与联盟起源","学术界最新冷战研究观点","二战后国际条约与安全体系","反冷战学派的观点","“新冷战”风险预测","美苏与当今中美对比","军备竞赛开支与经济数据","冷战对全球第三世界国家的影响","冷战与古代帝国对抗的类比","时间线梳理 / 史料争议 / 因果链分析"],
  "slots": [
    {"slot":"BIO","queries":["冷战关键人物与联盟起源"]},
    {"slot":"CURRENT","queries":["学术界最新冷战研究观点"]},
    {"slot":"RULES","queries":["二战后国际条约与安全体系"]},
    {"slot":"OPPOSITE","queries":["反冷战学派的观点"]},
    {"slot":"FUTURE","queries":["“新冷战”风险预测"]},
    {"slot":"COMPARISON","queries":["美苏与当今中美对比"]},
    {"slot":"DATA","queries":["军备竞赛开支与经济数据"]},
    {"slot":"IMPACT","queries":["冷战对全球第三世界国家的影响"]},
    {"slot":"WILDCARD","queries":["冷战与古代帝国对抗的类比"]},
    {"slot":"History","queries":["时间线梳理 / 史料争议 / 因果链分析"]}
  ]
}


"""


class QueryMaker:
    """
    QueryMaker generates diverse search queries using LLM
    to improve information retrieval coverage and quality.
    """
    
    def __init__(self, config: Optional[QueryMakerConfig] = None):
        """
        Initialize QueryMaker with configuration.
        
        Args:
            config: QueryMaker configuration. If None, loads from YAML.
        """
        self.config = config or self._load_config_from_yaml()
        self.client = self._init_openai_client()
        
        logger.info(f"QueryMaker initialized with model: {self.config.model}, num_queries: {self.config.num_queries}")
    
    def _load_config_from_yaml(self) -> QueryMakerConfig:
        """Load configuration from YAML config file"""
        cfg = get_config()
        
        api_base = cfg.get('external_services.openai.api_base')
        api_key = os.getenv("OPENAI_API_KEY_AGENT")
        model = cfg.get('models.querymaker.model_name', 'gpt-oss-20b')
        temperature = cfg.get('models.querymaker.temperature', 0.7)
        max_tokens = cfg.get('models.querymaker.max_tokens', 500)
        num_queries = cfg.get('models.querymaker.num_queries', 5)
        
        if not all([api_base, api_key]):
            raise ValueError("Missing required API configuration for QueryMaker")
        
        return QueryMakerConfig(
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_queries=num_queries
        )
    
    def _init_openai_client(self) -> openai.OpenAI:
        """Initialize OpenAI client"""
        try:
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=30
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _analyze_query_type(self, query: str) -> dict:
        """
        Analyze query type to determine optimal number of queries.
        
        Returns:
            dict with query type flags and recommended query count
        """
        query_lower = query.lower()
        query_stripped = query.strip()
        
        # Simple query detection (懒人模式：只输入一个名词/名字)
        # 特征：很短、没有问号、没有动词、单个词或短语
        is_simple = (
            len(query_stripped) <= 20 and  # 短查询
            '?' not in query_stripped and '？' not in query_stripped and  # 没有问号
            not any(keyword in query_lower for keyword in [
                '什么', '如何', '怎么', '为什么', '哪', '多少',
                'what', 'how', 'why', 'when', 'where', 'which', 'who'
            ]) and  # 没有疑问词
            len(query_stripped.split()) <= 4  # 词数很少（英文）或字数很少（中文）
        )
        
        # Definition queries (什么是, 是什么, what is, what are, 定义)
        is_definition = any(keyword in query_lower for keyword in [
            '什么是', '是什么', 'what is', 'what are', '定义', 'definition',
            '何为', '何謂', 'define', '概念'
        ])
        
        # Comparison queries (对比, 比较, compare, vs)
        is_comparison = any(keyword in query_lower for keyword in [
            '对比', '比较', 'compare', 'comparison', 'vs', 'versus', '差异', '區別'
        ])
        
        # How-to queries (如何, 怎么, how to)
        is_howto = any(keyword in query_lower for keyword in [
            '如何', '怎么', '怎样', 'how to', 'how can', 'how do', '怎樣'
        ])
        
        # Determine recommended query count (priority order matters!)
        # Comparison and How-to should be detected before Simple
        if is_comparison:
            recommended = 8  # Comparison: need multiple angles (highest priority)
        elif is_howto:
            recommended = 5  # How-to: methods + steps + examples
        elif is_definition:
            recommended = 2  # Definition: original + 1 variant
        elif is_simple:
            recommended = 2  # Simple: original + 1 enriched query (添加"是什么"/"介绍")
        else:
            recommended = self.config.num_queries  # Default from config
        
        return {
            'is_simple': is_simple,
            'is_definition': is_definition,
            'is_comparison': is_comparison,
            'is_howto': is_howto,
            'recommended_queries': recommended
        }
    
    def generate_queries(self, original_query: str, ctx: Optional[Context] = None) -> List[str]:
        """
        Generate diverse search queries based on the original query.
        
        Args:
            original_query: The user's original query
            ctx: Optional context containing time information, etc.
            
        Returns:
            List of diverse search queries (includes original query as first item)
        """
        try:
            # Analyze query type to determine optimal query count
            analysis = self._analyze_query_type(original_query)
            target_queries = analysis['recommended_queries']
            
            logger.info(f"[QueryMaker] Query type: simple={analysis['is_simple']}, "
                       f"def={analysis['is_definition']}, comp={analysis['is_comparison']}, "
                       f"howto={analysis['is_howto']}, target={target_queries}")
            
            if analysis['is_simple']:
                print(f"[QueryMaker][SIMPLE] Detected simple query (lazy mode): '{original_query}' - using {target_queries} queries")
            
            logger.info(f"[QueryMaker] Generating {target_queries} queries for: {original_query[:100]}...")
            
            # Get current time dynamically
            import datetime
            now = datetime.datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            current_year = str(now.year)
            current_month = now.strftime("%B %Y")  # e.g., "October 2025"
            next_year = str(now.year + 1)
            
            # Build prompt with time context
            # Use replace() instead of format() to avoid conflicts with JSON braces in examples
            prompt_with_time = QUERYMAKER_PROMPT.replace("{current_date}", current_date)
            prompt_with_time = prompt_with_time.replace("{current_year}", current_year)
            prompt_with_time = prompt_with_time.replace("{current_month}", current_month)
            prompt_with_time = prompt_with_time.replace("{next_year}", next_year)
            
            prompt = prompt_with_time + f"\n\n# Constraint: Limit total generated queries to about {target_queries}."
            
            print(f"[QueryMaker][DEBUG] Current time context: {current_date} (Year: {current_year})")
            
            # Call LLM
            print(f"[QueryMaker][DEBUG] Calling LLM with model={self.config.model}, temp={self.config.temperature}")
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": original_query}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            print(f"[QueryMaker][DEBUG] Response finish_reason: {response.choices[0].finish_reason}")
            
            # Handle both regular and reasoning models (o1 series)
            message = response.choices[0].message
            response_text = message.content
            
            # For reasoning models (o1), content might be None and reasoning_content has the text
            if response_text is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                response_text = message.reasoning_content
                logger.info(f"[QueryMaker] Using reasoning_content from o1 model")
            
            # Check if response is still None
            if response_text is None:
                logger.warning(f"[QueryMaker] LLM returned None response, finish_reason={response.choices[0].finish_reason}")
                return self._fallback_queries(original_query, ctx)
            
            response_text = response_text.strip()
            
            print(f"[QueryMaker][DEBUG] Raw response length: {len(response_text)}")
            print(f"[QueryMaker][DEBUG] Raw response (first 500 chars): {response_text[:500]}")
            
            # Try to extract JSON from response (handle cases where LLM adds extra text)
            json_text = response_text
            
            # If response contains markdown code blocks, extract JSON from them
            if '```json' in response_text:
                import re
                match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if match:
                    json_text = match.group(1)
                    print(f"[QueryMaker][DEBUG] Extracted JSON from markdown code block")
            elif '```' in response_text:
                import re
                match = re.search(r'```\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if match:
                    json_text = match.group(1)
                    print(f"[QueryMaker][DEBUG] Extracted JSON from code block")
            
            # Try to find JSON object by looking for { ... }
            if not json_text.startswith('{'):
                import re
                match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if match:
                    json_text = match.group(1)
                    print(f"[QueryMaker][DEBUG] Extracted JSON by pattern matching")
            
            # Parse JSON response
            try:
                parsed = json.loads(json_text)
                
                # Handle new structured format with slots
                if isinstance(parsed, dict) and 'slots' in parsed:
                    # Extract queries from all slots
                    valid_queries = []
                    for slot in parsed.get('slots', []):
                        slot_queries = slot.get('queries', [])
                        valid_queries.extend([q.strip() for q in slot_queries if isinstance(q, str) and q.strip()])
                    
                    logger.info(f"[QueryMaker] Extracted {len(valid_queries)} queries from {len(parsed.get('slots', []))} slots")
                
                # Handle simple list format (backward compatibility)
                elif isinstance(parsed, list):
                    valid_queries = [q.strip() for q in parsed if isinstance(q, str) and q.strip()]
                
                else:
                    logger.warning(f"[QueryMaker] Unexpected response format: {type(parsed)}, keys: {parsed.keys() if isinstance(parsed, dict) else 'N/A'}")
                    return self._fallback_queries(original_query, ctx)
                
                if not valid_queries:
                    logger.warning(f"[QueryMaker] No valid queries generated")
                    return self._fallback_queries(original_query, ctx)
                
                # Ensure we have the original query as first item
                if original_query not in valid_queries:
                    valid_queries = [original_query] + valid_queries
                else:
                    # Move original query to front
                    valid_queries.remove(original_query)
                    valid_queries = [original_query] + valid_queries
                
                # Limit to target number (based on query type)
                final_queries = valid_queries[:target_queries]
                
                logger.info(f"[QueryMaker] Generated {len(final_queries)} queries: {final_queries}")
                return final_queries
                
            except json.JSONDecodeError as e:
                logger.error(f"[QueryMaker] JSON parse error: {e}")
                print(f"[QueryMaker][ERROR] JSON parse failed at position {e.pos}")
                print(f"[QueryMaker][ERROR] Attempted to parse: {json_text[:300]}")
                print(f"[QueryMaker][ERROR] Full response length: {len(response_text)}")
                return self._fallback_queries(original_query, ctx)
        
        except Exception as e:
            logger.error(f"[QueryMaker] Failed to generate queries: {e}")
            print(f"[QueryMaker][ERROR] Exception type: {type(e).__name__}")
            print(f"[QueryMaker][ERROR] Exception details: {str(e)}")
            import traceback
            print(f"[QueryMaker][ERROR] Traceback:")
            traceback.print_exc()
            return self._fallback_queries(original_query, ctx)
    
    def _fallback_queries(self, original_query: str, ctx: Optional[Context] = None) -> List[str]:
        """
        Fallback query generation using simple heuristics.
        
        Args:
            original_query: The original query
            ctx: Optional context
            
        Returns:
            List containing the original query and simple variations
        """
        # Analyze query type for fallback as well
        analysis = self._analyze_query_type(original_query)
        target_queries = analysis['recommended_queries']
        
        queries = [original_query]
        
        # For simple queries (lazy mode), add one enriched query
        if analysis['is_simple']:
            # 检测语言并添加合适的扩展查询
            if any('\u4e00' <= c <= '\u9fff' for c in original_query):
                # 中文：添加"介绍"或"是什么"
                queries.append(f"{original_query} 介绍")
            else:
                # 英文：添加"overview"或"introduction"
                queries.append(f"{original_query} overview")
            logger.info(f"[QueryMaker][FALLBACK] Using {len(queries)} queries for simple query")
            return queries
        
        # For definition queries, keep it minimal
        if analysis['is_definition']:
            # Just return original query for definitions in fallback
            logger.info(f"[QueryMaker][FALLBACK] Using 1 query for definition type")
            return queries
        
        # Add time-based variation if time context exists
        if ctx and ctx.time_context and ctx.time_context.window[0]:
            date_str = ctx.time_context.window[0].split('T')[0]
            queries.append(f"{original_query} {date_str}")
        
        # Add "latest" variation
        if "最新" not in original_query and "latest" not in original_query.lower():
            queries.append(f"{original_query} latest")
        
        logger.info(f"[QueryMaker][FALLBACK] Using {len(queries)} fallback queries (target: {target_queries})")
        return queries[:target_queries]


# Convenience function for direct usage
def generate_search_queries(original_query: str, ctx: Optional[Context] = None, num_queries: int = 5) -> List[str]:
    """
    Convenience function to generate search queries.
    
    Args:
        original_query: The user's original query
        ctx: Optional context
        num_queries: Number of queries to generate
        
    Returns:
        List of diverse search queries
    """
    querymaker = QueryMaker()
    return querymaker.generate_queries(original_query, ctx) 
"""
Information Extractor Module
Extracts structured information from content passages using LLM and fallback methods.
"""

import os
import json
import re
import time
import asyncio
import hashlib
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field, replace

from config_manager import get_config

# Import from planner module
from planner import PlanTask, ExtractionPlan

# Import from ranker module
from actions.ranker import ContentPassage

# Avoid circular imports for type hints only
if TYPE_CHECKING:
    from actions.ir_rag import SearchResult

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ExtractedVariable:
    """Extracted information for a specific variable"""
    variable_name: str
    value: Any
    confidence: float
    provenance: List[str] = field(default_factory=list)  # Source URLs
    extraction_method: str = "llm"  # llm, regex, ner
    raw_passages: List[str] = field(default_factory=list)


# ============================================================================
# InformationExtractor Class
# ============================================================================

class InformationExtractor:
    """Component for extracting structured information from passages"""

    def __init__(self, llm_client=None, model_name: str = None):
        self.client = llm_client
        cfg = get_config()
        self.model_name = model_name or os.getenv("OPENAI_API_MODEL_AGENT_SYNTHESIZER", "gpt-3.5-turbo")
        self.small_model_name = cfg.get('models.extractor.small_model_name', self.model_name)
        self.small_max_tokens = cfg.get('models.extractor.small_max_tokens', 1000)
        self.small_temperature = cfg.get('models.extractor.small_temperature', 0.0)
        default_max_tokens = cfg.get('models.extractor.max_tokens', 4000)
        self.stage2_max_tokens = cfg.get('models.extractor.stage2_max_tokens', max(1500, default_max_tokens // 2))
        self.stage2_chunk_limit = cfg.get('ir_rag.extraction.stage2_chunk_limit', 3)
        self.snippet_confidence_threshold = cfg.get('ir_rag.extraction.snippet_confidence_threshold', 0.75)
        self.stage1_concurrency = cfg.get('ir_rag.extraction.stage1_concurrency', 5)
        self.stage2_concurrency = cfg.get('ir_rag.extraction.stage2_concurrency', 3)
        self._cache: Dict[str, ExtractedVariable] = {}

    def _looks_structured(self, text: str) -> bool:
        """Heuristic, format-agnostic detection of structure (markdown/html tables or explicit [TABLE]/[LIST])."""
        if not text:
            return False
        t = text.lower()
        # explicit structured markers or common table/list signals
        return (
            "[table]" in t or "[list]" in t or
            "<table" in t or "&lt;table" in t or
            "\n|" in text or "| " in text or
            re.search(r'^\s*[-*]\s+\S', text, flags=re.MULTILINE) is not None
        )

    def _pick_body_text(self, passage: 'ContentPassage', meta: Dict[str, Any], mode: str) -> str:
        """
        Prefer structure-preserving text when available.
        - If mode == 'snippet' and meta['snippet'] exists, use it (for SERP snippets).
        - If mode == 'mixed': use raw_text if it has structure, else use snippet if available.
        - Else prefer passage.raw_text (keeps tables/lists), fallback to passage.text.
        """
        if mode == "snippet" and meta.get('snippet') and passage.source_url == "SERP_SNIPPET":
            return str(meta['snippet'])
        
        if mode == "mixed":
            # For mixed mode: prioritize structured content from raw_text
            raw = passage.raw_text or ""
            if raw and self._looks_structured(raw):
                return raw
            # Fallback to snippet for SERP results, or text for others
            if passage.source_url == "SERP_SNIPPET" and meta.get('snippet'):
                return str(meta['snippet'])
            return raw or passage.text or ""
        
        return passage.raw_text or passage.text or ""

    def _summarize_for_cache(self, s: str) -> str:
        """Stable short fingerprint source for cache/dedupe (avoid huge keys)."""
        if not s:
            return ""
        s = s.strip()
        if len(s) <= 200:
            return s
        return s[:100] + s[-100:]
    
    async def extract_variables(
        self,
        plan: ExtractionPlan,
        ranked_passages: Dict[str, List[ContentPassage]],
        search_results: Optional[List['SearchResult']] = None,
        ctx: Optional[Any] = None
    ) -> Dict[str, ExtractedVariable]:
        """Two-stage extraction using fast snippets first, then deep passages."""

        task_count = len(plan.tasks_to_extract)
        print(f"[EXTRACTOR][Pipeline] Starting two-stage extraction for {task_count} variables")

        extracted_vars: Dict[str, ExtractedVariable] = {}
        unresolved: List[tuple] = []  # (task, passages, stage1_candidate)

        stage1_start = time.perf_counter()

        # Run Stage 1 concurrently for all tasks
        stage1_results = await self._run_stage1(plan.tasks_to_extract, ranked_passages, ctx)

        stage1_solved = 0
        for task, passages, stage1_result in stage1_results:
            if stage1_result:
                extracted_vars[task.variable_name] = stage1_result
                if stage1_result.confidence >= self.snippet_confidence_threshold:
                    stage1_solved += 1
                    print(
                        f"[EXTRACTOR][Stage1] Satisfied '{task.variable_name}' "
                        f"(confidence={stage1_result.confidence:.2f})"
                    )
                    continue
                else:
                    print(
                        f"[EXTRACTOR][Stage1] Tentative '{task.variable_name}' "
                        f"(confidence={stage1_result.confidence:.2f})"
                    )

            unresolved.append((task, passages, stage1_result))

        stage1_elapsed = time.perf_counter() - stage1_start
        print(
            f"[EXTRACTOR][Stage1] Solved {stage1_solved}/{task_count} variables "
            f"in {stage1_elapsed:.2f}s (concurrent)"
        )

        if not unresolved:
            return extracted_vars

        stage2_results = await self._run_stage2(unresolved, ctx)

        for task, result in stage2_results:
            prior = extracted_vars.get(task.variable_name)
            if prior is None or (result and result.confidence >= prior.confidence):
                extracted_vars[task.variable_name] = result

        threshold_map = {t.variable_name: t.confidence_threshold for t in plan.tasks_to_extract}
        solved_total = sum(
            1
            for name, value in extracted_vars.items()
            if value and value.confidence >= threshold_map.get(name, self.snippet_confidence_threshold)
        )
        print(
            f"[EXTRACTOR][Pipeline] Stage2 resolved {len(stage2_results)} tasks; "
            f"total confident extractions: {solved_total}/{task_count}"
        )

        return extracted_vars

    async def _run_stage1(
        self,
        tasks: List[PlanTask],
        ranked_passages: Dict[str, List[ContentPassage]],
        ctx: Optional[Any] = None
    ) -> List[tuple]:
        """
        Run Stage 1 extraction using batch mode for efficiency.
        Extracts all tasks in one LLM call with shared passages.
        Returns list of (task, passages, stage1_result) tuples.
        """
        if not tasks:
            return []

        # Try batch extraction first (more efficient)
        try:
            batch_results = await self._run_stage1_batch(tasks, ranked_passages, ctx)
            if batch_results:
                return batch_results
        except Exception as e:
            print(f"[EXTRACTOR][Stage1] Batch mode failed: {e}, falling back to individual extraction")
        
        # Fallback to individual extraction if batch fails
        return await self._run_stage1_individual(tasks, ranked_passages, ctx)
    
    async def _run_stage1_batch(
        self,
        tasks: List[PlanTask],
        ranked_passages: Dict[str, List[ContentPassage]],
        ctx: Optional[Any] = None
    ) -> List[tuple]:
        """
        Batch extraction: extract all tasks in one LLM call.
        More efficient and provides global context for deduplication.
        """
        if not tasks or not self.client:
            return None

        # Step 1: Merge and deduplicate passages from all tasks
        all_passages_dict = {}
        for task in tasks:
            passages = ranked_passages.get(task.variable_name, [])
            for p in passages:
                # Use URL as key for deduplication
                key = f"{p.source_url}::{p.text[:100]}"  # Include text prefix for better deduplication
                if key not in all_passages_dict or p.score > all_passages_dict[key].score:
                    all_passages_dict[key] = p

        # Step 2: Filter to snippets only and sort by score
        snippet_passages = [
            p for p in all_passages_dict.values()
            if p.source_url == "SERP_SNIPPET"
        ]
        snippet_passages.sort(key=lambda p: p.score, reverse=True)

        # Step 3: Limit to top snippets to avoid token overflow
        top_snippets = snippet_passages[:15]  # Top 15 most relevant snippets

        # Optionally add a couple of structure-rich passages (tables/lists) to help roster/list tasks.
        struct_candidates = [
            p for p in all_passages_dict.values()
            if p.source_url != "SERP_SNIPPET" and self._looks_structured(p.raw_text or p.text)
        ]
        struct_candidates.sort(key=lambda p: p.score, reverse=True)
        top_struct = struct_candidates[:2]

        shared_inputs = top_snippets + top_struct

        if not shared_inputs:
            print(f"[EXTRACTOR][Stage1-Batch] No passages available for batch extraction")
            return None

        print(f"[EXTRACTOR][Stage1-Batch] Extracting {len(tasks)} tasks from {len(shared_inputs)} shared passages (snippets + structured)")

        # Step 4: Batch extract all tasks
        batch_results = await self._batch_extract(tasks, shared_inputs, ctx, stage="stage1")

        # Step 5: Map results back to (task, passages, result) format
        results = []
        for task in tasks:
            passages = ranked_passages.get(task.variable_name, [])
            result = batch_results.get(task.variable_name)
            results.append((task, passages, result))

        return results
    
    async def _run_stage1_individual(
        self,
        tasks: List[PlanTask],
        ranked_passages: Dict[str, List[ContentPassage]],
        ctx: Optional[Any] = None
    ) -> List[tuple]:
        """
        Original individual extraction logic as fallback.
        """
        semaphore = asyncio.Semaphore(max(1, self.stage1_concurrency))
        
        async def process_task(task: PlanTask) -> tuple:
            passages = ranked_passages.get(task.variable_name, [])
            if not passages:
                print(f"[EXTRACTOR][Stage1] No passages for {task.variable_name}")
                return task, passages, None

            snippet_passages = [p for p in passages if p.source_url == "SERP_SNIPPET"]
            stage1_result = None

            if snippet_passages and self.client:
                async with semaphore:
                    stage1_result = await self._stage1_extract(task, snippet_passages[:5], ctx)

            return task, passages, stage1_result

        # Process all tasks concurrently
        results = await asyncio.gather(*[process_task(task) for task in tasks])
        return results

    async def _stage1_extract(self, task: PlanTask, snippet_passages: List[ContentPassage], ctx: Optional[Any] = None) -> Optional[ExtractedVariable]:
        """Fast snippet-first extraction using a lightweight model."""
        try:
            result = await self._llm_extract(
                task,
                snippet_passages,
                stage_label="llm-snippet",
                model_name=self.small_model_name,
                max_tokens=self.small_max_tokens,
                temperature=self.small_temperature,
                mode="snippet",
                allow_fallback=False,
                ctx=ctx
            )
            return result
        except Exception as exc:
            logger.error(f"[EXTRACTOR][STAGE1] Failed for {task.variable_name}: {exc}")
            return None
    
    async def _batch_extract(
        self,
        tasks: List[PlanTask],
        passages: List[ContentPassage],
        ctx: Optional[Any] = None,
        stage: str = "stage1"
    ) -> Dict[str, ExtractedVariable]:
        """
        Batch extract multiple tasks from shared passages in one LLM call.
        More efficient and provides global context for automatic deduplication.
        """
        if not tasks or not passages or not self.client:
            return {}
        
        cfg = get_config()
        
        # Step 1: Format passages as structured EVIDENCE
        # Use mixed mode: prefer raw_text for structured content, snippet for SERP snippets
        mode = "mixed"  # Allow _pick_body_text to intelligently choose
        combined_text, provenance, raw_texts = self._format_passages(passages, mode=mode)
        
        if not combined_text.strip():
            return {}
        
        # Step 2: Build PLAN_TASKS JSON (all tasks)
        plan_tasks_json = json.dumps([
            {
                "variable_name": task.variable_name,
                "fact": task.fact,
                "category": task.category
            }
            for task in tasks
        ], ensure_ascii=False)
        
        # Step 3: Detect language
        lang = self._detect_language(tasks[0].fact + " " + combined_text[:500])
        
        # Step 4: Extract time context
        time_rules = ""
        if ctx and hasattr(ctx, 'time_context') and ctx.time_context:
            tc = ctx.time_context
            if tc.window and tc.window[0]:
                target_date = tc.window[0].split('T')[0]
                if lang == "zh":
                    time_rules = f"\n\n时间规则：目标日期 = {target_date}。只接受明确标注此日期的数据。"
                else:
                    time_rules = f"\n\nTime Rule: Target date = {target_date}. Only accept data explicitly marked with this date."
        
        # Step 5: Build prompt
        if lang == "zh":
            system_msg = """你是结构化提取专家。从EVIDENCE中提取所有PLAN_TASKS要求的事实。
输出严格JSON，不猜测。如缺失或冲突，标注"status":"unverified"或"conflict"。

批量提取规则：
- 为每个task返回一个result
- 从所有EVIDENCE中找到最相关的信息
- 如果多个tasks语义重复（如achievements和notable_facts），合并到一个task中，其他返回null
- 如果某task没有信息，返回"status":"unverified", "value":null

列表/名单/表格提取规则（关键）：
- 当task要求提取名单、列表、成员、领导人、官员时，必须提取所有名字，不要仅总结
- 寻找模式："XXX局长：YYY"、"XXX: YYY (年份)"、表格格式[TABLE]...[/TABLE]
- 如果看到[TABLE]标记，提取**所有行**的数据
- **年份是可选的**：如果表格/文本中有年份就附带，没有年份也照样提取姓名+职位
- 格式化为结构化列表：
  * 有年份："职位1：姓名1（2022年7月）; 职位2：姓名2（2020年）; ..."
  * 无年份："职位1：姓名1; 职位2：姓名2; ..."
  * 混合："职位1：姓名1（2022年）; 职位2：姓名2; 职位3：姓名3（2023年）; ..."
- 示例：
  ✅ 正确："政務司司長：陳國基（2022年7月）; 財政司司長：陳茂波（2017年1月）; 律政司司長：林定國（2022年7月）"
  ✅ 正确："公務員事務局：楊何蓓茵; 政制及內地事務局：曾國衞; 文化體育及旅遊局：羅淑佩"（没有年份也可以）
  ❌ 错误："有三位司長"（缺少具体名字）
  ❌ 错误："包括政務司、財政司、律政司"（只有职位，缺少人名）
- 即使文本很长，也要提取所有提到的名字，不要遗漏
- **重要**：即使任务名包含 "with_years"，也要提取没有年份的条目

时间信息提取规则（仅适用于非列表类任务）：
- 对于单一事件/奖项/成就（非列表），尽量提取年份
- 年份必须与事件明确关联（同句或邻近±120字符）
- 忽略页面元数据日期（"更新于"、"发布于"）
- 职业/奖项格式："Position (YYYY)" 或 "Position, started YYYY"
- 如果找不到年份但事实存在，value正常填写，notes中说明"年份未找到"
- 区分事件类型：当选/任命/就职/卸任/获奖，每个都要标注年份
- **注意**：对于列表/名单类任务，年份是可选的（见上方列表提取规则）
"""
            prompt = f"""INPUT
PLAN_TASKS: {plan_tasks_json}
EVIDENCE: {combined_text}{time_rules}

OUTPUT JSON SCHEMA
{{
  "results": [
    {{
      "variable_name": "...", 
      "status": "ok"|"unverified"|"conflict",
      "value": "string (if temporal: include year as 'Position (YYYY)' or '事件 (YYYY年)')",
      "source_ids": [1,2],
      "notes": "简短原因，如涉及时间但未找到年份，必须说明"
    }},
    ...
  ]
}}

要求：
1. 为PLAN_TASKS中的每个task返回一个result
2. 从EVIDENCE中选择最相关的信息
3. 特别注意：如果task涉及职位/奖项/事件，value中必须包含年份信息
4. 如果发现tasks重复，优先填充第一个，其他返回null
5. 仅返回JSON，不要解释。"""
        else:
            system_msg = """You are a Structured Extractor. Extract all PLAN_TASKS from EVIDENCE in one batch.
Output STRICT JSON only. Do NOT guess. If missing or conflicting, mark "status":"unverified" or "conflict".

Batch extraction rules:
- Return one result for each task
- Find the most relevant information from all EVIDENCE
- If multiple tasks are semantically duplicate (e.g. achievements and notable_facts), merge into one task, return null for others
- If no information for a task, return "status":"unverified", "value":null

List/Roster/Table Extraction Rules (CRITICAL):
- When task asks for lists, rosters, members, leaders, officials: extract ALL names, do NOT just summarize
- Look for patterns: "XXX Chief: YYY", "XXX: YYY (Year)", table formats [TABLE]...[/TABLE]
- If you see [TABLE] markers, extract **ALL rows** of data
- **Year is OPTIONAL**: If table/text has year info, include it; if no year, still extract name+position
- Format as structured list:
  * With year: "Position1: Name1 (July 2022); Position2: Name2 (2020); ..."
  * Without year: "Position1: Name1; Position2: Name2; ..."
  * Mixed: "Position1: Name1 (2022); Position2: Name2; Position3: Name3 (2023); ..."
- Examples:
  ✅ GOOD: "Chief Secretary: Chan Kwok-ki (July 2022); Financial Secretary: Paul Chan (Jan 2017); Secretary for Justice: Paul Lam (July 2022)"
  ✅ GOOD: "Civil Service Bureau: Ada Chung; Constitutional Affairs Bureau: Erick Tsang; Culture Bureau: Rosanna Law" (no year is OK)
  ❌ BAD: "There are three secretaries" (missing actual names)
  ❌ BAD: "Including Chief Secretary, Financial Secretary, and Secretary for Justice" (titles only, missing names)
- Even if text is long, extract ALL mentioned names, do not omit any
- **IMPORTANT**: Even if task name contains "with_years", extract entries WITHOUT year too

Temporal Information Extraction Rules (for non-list tasks only):
- For single events/awards/achievements (non-list), try to extract year
- Year must be explicitly linked to the event (same sentence or within ±120 chars)
- Ignore page meta dates ("updated on", "published on", "last modified")
- Format for positions/awards: "Position (YYYY)" or "Position, started YYYY"
- If year not found but fact exists, fill value normally and note "year not found" in notes
- Distinguish event types: elected/appointed/assumed/resigned/awarded, each with year
- **Note**: For list/roster tasks, year is OPTIONAL (see list extraction rules above)
"""
            prompt = f"""INPUT
PLAN_TASKS: {plan_tasks_json}
EVIDENCE: {combined_text}{time_rules}

OUTPUT JSON SCHEMA
{{
  "results": [
    {{
      "variable_name": "...", 
      "status": "ok"|"unverified"|"conflict",
      "value": "string (if temporal: include year as 'Position (YYYY)' or 'Event in YYYY')",
      "source_ids": [1,2],
      "notes": "short reason, if temporal but year not found, must explain"
    }},
    ...
  ]
}}

Requirements:
1. Return one result for each task in PLAN_TASKS
2. Select most relevant information from EVIDENCE
3. CRITICAL: If task involves position/award/event, value MUST include year information
4. If tasks are duplicate, prioritize first one, return null for others
5. Return JSON only, no explanation."""
        
        # Step 6: Call LLM
        model_name = self.small_model_name if stage == "stage1" else self.model_name
        max_tokens = (self.small_max_tokens * 2) if stage == "stage1" else (self.stage2_max_tokens * 2)
        
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic for batch extraction
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
        except Exception as api_error:
            print(f"[EXTRACTOR][BATCH] response_format not supported, retrying without it")
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=max_tokens
            )
        
        # Step 7: Parse results
        result_text = (response.choices[0].message.content or "").strip()
        if not result_text:
            print(f"[EXTRACTOR][BATCH] Empty LLM response")
            return {}
        
        try:
            result_data = json.loads(result_text)
        except json.JSONDecodeError as err:
            print(f"[EXTRACTOR][BATCH] JSON decode failed: {err}")
            print(f"[EXTRACTOR][BATCH] Raw response: {result_text[:200]}")
            return {}
        
        # Step 8: Map results back to ExtractedVariable format
        extracted_vars = {}
        results = result_data.get("results", [])
        
        for result in results:
            var_name = result.get("variable_name")
            if not var_name:
                continue
            
            status = result.get("status", "ok")
            value = result.get("value")
            notes = result.get("notes", "")
            
            # Calculate confidence based on status
            if status == "ok":
                confidence = 0.95
            elif status == "unverified":
                confidence = 0.60
            elif status == "conflict":
                confidence = 0.40
            else:
                confidence = 0.50
            
            # 🔥 FALLBACK: If extraction failed, use raw material
            extraction_method = f"batch-{stage}"
            if (not value or confidence < 0.7) and raw_texts:
                # Find the corresponding task for better context
                task_obj = next((t for t in tasks if t.variable_name == var_name), None)
                if task_obj:
                    raw_material = self._extract_relevant_raw_material(raw_texts, task_obj)
                    if raw_material and len(raw_material) > 50:
                        value = raw_material
                        confidence = 0.65
                        status = "raw_fallback"
                        notes = f"批量提取失败，返回原始材料。原因：{notes}"
                        extraction_method = "batch-raw-fallback"
                        print(f"[EXTRACTOR][BATCH-FALLBACK] {var_name}: using raw material ({len(raw_material)} chars)")
            
            # Map source_ids to provenance URLs
            source_ids = result.get("source_ids", [])
            selected_provenance = [provenance[i-1] for i in source_ids if 1 <= i <= len(provenance)]
            if not selected_provenance:
                selected_provenance = provenance[:3]  # Use first 3 as fallback
            
            extracted_vars[var_name] = ExtractedVariable(
                variable_name=var_name,
                value=value,
                confidence=confidence,
                provenance=selected_provenance,
                extraction_method=extraction_method,
                raw_passages=raw_texts[:3]  # Include sample passages
            )
            
            if cfg.is_decision_logging_enabled('ir_rag'):
                value_preview = str(value)[:50] if value else 'None'
                print(
                    f"[EXTRACTOR][BATCH] {var_name}: value={value_preview}... "
                    f"confidence={confidence:.2f} status={status}"
                )
        
        print(f"[EXTRACTOR][BATCH] Successfully extracted {len(extracted_vars)}/{len(tasks)} tasks")
        
        return extracted_vars

    async def _run_stage2(self, unresolved: List[tuple], ctx: Optional[Any] = None) -> List[tuple]:
        """Run focused deep extraction for unresolved variables concurrently."""
        if not unresolved:
            return []

        stage2_start = time.perf_counter()

        if not self.client:
            logger.warning("[EXTRACTOR][Stage2] No LLM client; using fallback extraction")
            results = []
            for task, passages, _ in unresolved:
                fallback = self._fallback_extract(task, passages)
                results.append((task, fallback))
            return results

        semaphore = asyncio.Semaphore(max(1, self.stage2_concurrency))
        tasks = [self._stage2_extract(task, passages, provisional, semaphore, ctx) for task, passages, provisional in unresolved]
        stage2_outputs = await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - stage2_start
        solved = sum(1 for _, result in stage2_outputs if result and result.confidence >= 0.6)
        print(
            f"[EXTRACTOR][Stage2] Completed {len(stage2_outputs)} tasks in {elapsed:.2f}s "
            f"(confidence ≥0.6: {solved}, concurrent)"
        )

        return stage2_outputs

    async def _stage2_extract(
        self,
        task: PlanTask,
        passages: List[ContentPassage],
        provisional: Optional[ExtractedVariable],
        semaphore: asyncio.Semaphore,
        ctx: Optional[Any] = None
    ) -> tuple:
        """Run deep extraction with focused passages and caching."""

        focused_passages = self._prepare_stage2_passages(passages)
        cache_key = self._cache_key(task, focused_passages, "llm-deep")

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            print(f"[EXTRACTOR][Stage2] Cache hit for {task.variable_name}")
            return task, replace(cached)

        focused_passages = focused_passages[:2000]
        async with semaphore:
            try:
                result = await self._llm_extract(
                    task,
                    focused_passages,
                    stage_label="llm-deep",
                    model_name=self.model_name,
                    max_tokens=self.stage2_max_tokens,
                    temperature=0.1,
                    mode="passage",
                    allow_fallback=True,
                    ctx=ctx
                )
                if result:
                    self._cache[cache_key] = result
                    return task, result
            except Exception as exc:
                logger.error(f"[EXTRACTOR][STAGE2] Failed for {task.variable_name}: {exc}")

        # Fallback if we reach here
        if provisional and provisional.confidence > 0:
            return task, provisional
        return task, self._fallback_extract(task, focused_passages or passages)

    def _prepare_stage2_passages(self, passages: List[ContentPassage]) -> List[ContentPassage]:
        """Select top passages from fetched pages and keep supporting snippets."""
        if not passages:
            return []

        primary = [p for p in passages if p.source_url != "SERP_SNIPPET"]
        secondary = [p for p in passages if p.source_url == "SERP_SNIPPET"]

        primary.sort(key=lambda p: p.score, reverse=True)
        selected = primary[: self.stage2_chunk_limit]

        if len(selected) < self.stage2_chunk_limit and secondary:
            need = self.stage2_chunk_limit - len(selected)
            selected.extend(secondary[:need])

        # Deduplicate by structure-preserving text (prefer raw_text), compact
        seen_text = set()
        unique_passages = []
        for passage in selected:
            base = passage.raw_text or passage.text or ""
            text_key = self._summarize_for_cache(base)
            if text_key in seen_text:
                continue
            seen_text.add(text_key)
            unique_passages.append(passage)

        return unique_passages

    def _format_passages(self, passages: List[ContentPassage], mode: str) -> tuple:
        """Create structured EVIDENCE list for entity-value extraction."""
        evidence_list = []
        provenance = []
        raw_texts = []

        for idx, passage in enumerate(passages, 1):
            meta = passage.metadata or {}
            title = meta.get('title') or meta.get('page_title') or ""
            provenance_id = meta.get('provenance', passage.source_url)
            snippet_date = meta.get('snippet_date')

            body_text = self._pick_body_text(passage, meta, mode)

            raw_texts.append(body_text)
            provenance.append(provenance_id)

            is_structured = self._looks_structured(body_text)

            # Keep body moderately bounded but preserve multi-line tables/lists
            # For structured content (tables), allow up to 5000 chars to avoid truncating large rosters
            if is_structured:
                snippet_body = body_text[:5000] if len(body_text) > 5000 else body_text
            else:
                snippet_body = body_text[:1000] if len(body_text) > 1000 else body_text
            evidence_item = {
                "id": idx,
                "url": provenance_id,
                "title": title,
                "snippet": snippet_body,
                "type": "snippet" if passage.source_url == "SERP_SNIPPET" else "webpage",
                "format": "structured" if is_structured else "plain"
            }
            if snippet_date:
                evidence_item["date"] = snippet_date

            evidence_list.append(evidence_item)

        # Format as JSON string for prompt
        evidence_json = json.dumps(evidence_list, ensure_ascii=False, indent=2)
        return evidence_json, provenance, raw_texts

    def _cache_key(self, task: PlanTask, passages: List[ContentPassage], stage_label: str) -> str:
        digest = hashlib.sha1()
        digest.update(task.variable_name.encode('utf-8'))
        digest.update(stage_label.encode('utf-8'))
        for passage in passages:
            base = passage.raw_text or passage.text or ""
            digest.update(self._summarize_for_cache(base).encode('utf-8', errors='ignore'))
        return digest.hexdigest()

    async def _extract_single_variable(self, task: PlanTask, passages: List[ContentPassage]) -> ExtractedVariable:
        """Backward-compatible single variable extraction."""
        try:
            if self.client:
                result = await self._llm_extract(task, passages, stage_label="llm-deep")
                if result:
                    return result
            return self._fallback_extract(task, passages)
        except Exception as e:
            logger.error(f"Extraction failed for {task.variable_name}: {e}")
            return self._fallback_extract(task, passages)
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Chinese or English"""
        if not text:
            return "en"
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.strip())
        if total_chars == 0:
            return "en"
        return "zh" if chinese_chars / total_chars > 0.3 else "en"
    
    def _is_list_task(self, task: PlanTask) -> bool:
        """Check if task requires list/roster extraction"""
        var_name_lower = task.variable_name.lower()
        fact_lower = task.fact.lower()
        
        # Keywords that indicate list extraction
        list_keywords = [
            'list', 'roster', 'chiefs', 'heads', 'members', 'leaders', 
            'names', 'officials', 'ministers', 'secretaries', 'directors',
            '名单', '局长', '司长', '成员', '领导', '官员', '部长'
        ]
        
        # Check variable name and fact
        for keyword in list_keywords:
            if keyword in var_name_lower or keyword in fact_lower:
                return True
        
        # Check category
        if task.category == "aggregation":
            # Additional check: fact asks for multiple entities
            if any(word in fact_lower for word in ['all', 'each', '所有', '各', '每个', 'who are']):
                return True
        
        return False
    
    def _direct_table_extract(self, task: PlanTask, passages: List[ContentPassage]) -> Optional[ExtractedVariable]:
        """
        Directly parse tables from passages for list/roster tasks.
        Bypass LLM to avoid uncertainty and conservative behavior.
        """
        all_entries = []
        provenance_urls = []
        
        for passage in passages:
            # Get raw text (which should contain [TABLE] markers)
            text = passage.raw_text or passage.text or ""
            
            if not text or '[TABLE]' not in text:
                continue
            
            # Extract table blocks
            table_blocks = re.findall(r'\[TABLE\](.*?)\[/TABLE\]', text, re.DOTALL)
            
            for table_text in table_blocks:
                entries = self._parse_table_block(table_text)
                if entries:
                    all_entries.extend(entries)
                    url = (passage.metadata or {}).get('provenance', passage.source_url)
                    if url not in provenance_urls:
                        provenance_urls.append(url)
        
        if not all_entries:
            print(f"[EXTRACTOR][DIRECT] No table entries found for {task.variable_name}")
            return None
        
        # Format entries as semicolon-separated list
        value = "; ".join(all_entries)
        
        # Calculate confidence based on number of entries
        if len(all_entries) >= 10:
            confidence = 0.95
        elif len(all_entries) >= 5:
            confidence = 0.90
        elif len(all_entries) >= 3:
            confidence = 0.85
        else:
            confidence = 0.75
        
        print(f"[EXTRACTOR][DIRECT] Extracted {len(all_entries)} entries from tables")
        
        return ExtractedVariable(
            variable_name=task.variable_name,
            value=value,
            confidence=confidence,
            provenance=provenance_urls,
            extraction_method="direct-table-parse",
            raw_passages=[text[:500] for text in [p.raw_text or p.text for p in passages if p.raw_text or p.text]]
        )
    
    def _parse_table_block(self, table_text: str) -> List[str]:
        """
        Parse a table block into list of entries.
        Expected format: "局名 | 局长姓名 | 副局长 | 政治助理"
        Returns: ["局名：局长姓名（年份）", ...] or ["局名：局长姓名", ...] if no year
        
        Flexible extraction: year is optional, extract entries even without year info.
        """
        entries = []
        lines = table_text.strip().split('\n')
        
        if not lines:
            return entries
        
        # Skip title/caption lines
        data_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('表格标题:') or line.startswith('---') or line == '------------':
                continue
            if '|' in line:
                data_lines.append(line)
        
        if not data_lines:
            return entries
        
        # First line might be header
        header_line = data_lines[0]
        headers = [h.strip() for h in header_line.split('|')]
        
        # Check if first line is header (contains generic terms)
        header_keywords = ['姓名', 'name', '职位', 'position', 'title', '年份', 'year', 'since', '局', 'bureau', 'department', 'from', 'appointed']
        is_header = any(any(kw in h.lower() for kw in header_keywords) for h in headers)
        
        start_idx = 1 if is_header else 0
        
        # Parse data rows
        for line in data_lines[start_idx:]:
            cells = [c.strip() for c in line.split('|')]
            
            if len(cells) < 2:
                continue
            
            # First cell is usually position/bureau, second is name
            position = cells[0]
            name = cells[1]
            
            # Skip if name is "不適用", "待定", empty, or generic placeholders
            skip_values = ['不適用', '不适用', 'N/A', '待定', 'TBD', 'TBA', '-', '']
            if not name or name in skip_values:
                continue
            
            # Also skip if name looks like a section header (all Chinese chars but very generic)
            if name in ['姓名', '局长', '司长', 'Name', 'Chief', 'Secretary']:
                continue
            
            # Try to find year/date in ALL cells (not just cells[2:])
            year = None
            date_info = None
            
            # Strategy 1: Look for 4-digit year (YYYY)
            for cell in cells:
                year_match = re.search(r'(20\d{2})', cell)
                if year_match:
                    year = year_match.group(1)
                    # Also try to find month
                    month_match = re.search(r'(20\d{2})[年-](\d{1,2})', cell)
                    if month_match:
                        date_info = f"{month_match.group(1)}年{month_match.group(2)}月"
                    else:
                        date_info = f"{year}年"
                    break
            
            # Strategy 2: Look for Chinese date patterns (like "2022年7月")
            if not date_info:
                for cell in cells:
                    cn_date_match = re.search(r'(20\d{2})年(\d{1,2})月', cell)
                    if cn_date_match:
                        date_info = f"{cn_date_match.group(1)}年{cn_date_match.group(2)}月"
                        break
            
            # Format entry based on available information
            # Priority: name + position (always) + date (if available)
            if date_info:
                entry = f"{position}：{name}（{date_info}）"
            elif year:
                entry = f"{position}：{name}（{year}年）"
            else:
                # No year info available, still extract name + position
                entry = f"{position}：{name}"
            
            entries.append(entry)
        
        return entries
    
    def _extract_relevant_raw_material(self, raw_texts: List[str], task: PlanTask) -> str:
        """
        Extract relevant raw material from passages when LLM extraction fails.
        Intelligently selects and formats the most relevant parts.
        """
        if not raw_texts:
            return ""
        
        # Extract keywords from task for relevance matching
        task_keywords = set()
        
        # From variable name (split by underscore)
        var_parts = task.variable_name.lower().replace('_', ' ').split()
        task_keywords.update([p for p in var_parts if len(p) > 3])
        
        # From fact (extract meaningful words)
        fact_words = re.findall(r'\b[\u4e00-\u9fff]{2,}|\b[a-zA-Z]{4,}\b', task.fact.lower())
        task_keywords.update(fact_words[:10])  # Limit to avoid too many
        
        # Score each raw text by keyword relevance
        scored_texts = []
        for text in raw_texts:
            if not text or len(text.strip()) < 50:
                continue
            
            text_lower = text.lower()
            score = 0
            
            # Count keyword matches
            for keyword in task_keywords:
                if keyword in text_lower:
                    score += 1
            
            # Bonus for structured content (tables/lists)
            if '[TABLE]' in text:
                score += 5
            if re.search(r'^\s*[-•*]\s', text, re.MULTILINE):
                score += 2
            
            scored_texts.append((score, text))
        
        if not scored_texts:
            # No scoring worked, just use first non-empty text
            for text in raw_texts:
                if text and len(text.strip()) > 50:
                    return self._format_raw_material(text[:1500])
            return ""
        
        # Sort by score (highest first) and take top passages
        scored_texts.sort(key=lambda x: x[0], reverse=True)
        
        # Combine top 2-3 passages
        selected = []
        total_chars = 0
        max_chars = 2000  # Limit total length
        
        for score, text in scored_texts[:3]:
            if score == 0:
                break  # No more relevant texts
            
            if total_chars + len(text) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 200:  # Only add if there's meaningful space left
                    selected.append(text[:remaining] + "...")
                break
            
            selected.append(text)
            total_chars += len(text)
        
        if not selected:
            # Fallback: use first passage
            return self._format_raw_material(raw_texts[0][:1500])
        
        # Combine and format
        combined = "\n\n---\n\n".join(selected)
        return self._format_raw_material(combined)
    
    def _format_raw_material(self, text: str) -> str:
        """
        Format raw material for better readability.
        Add a prefix to indicate it's raw content.
        """
        if not text:
            return ""
        
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Add indicator prefix
        formatted = f"[原始材料]\n{text.strip()}"
        
        return formatted
    
    def _extract_core_entities(self, question: str) -> List[str]:
        """
        Extract core entities from the question for validation.
        Simple heuristic-based extraction (can be enhanced with NER later).
        """
        # Remove common question words
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 
                         '什么', '何时', '哪里', '谁', '为什么', '怎么', '哪个', '多少']
        
        # Tokenize and filter
        import jieba
        words = list(jieba.cut(question))
        
        # Filter out question words, punctuation, and short words
        entities = []
        for word in words:
            word_lower = word.lower().strip()
            if (len(word) >= 2 and 
                word_lower not in question_words and 
                not word.strip() in '，。？！、；：""''（）【】《》' and
                not word.isdigit()):
                entities.append(word)
        
        # Return top entities (limit to avoid too many)
        return entities[:5]
    
    async def _llm_extract(
        self,
        task: PlanTask,
        passages: List[ContentPassage],
        stage_label: str = "llm",
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        mode: str = "passage",
        allow_fallback: bool = True,
        ctx: Optional[Any] = None
    ) -> Optional[ExtractedVariable]:
        """Use an LLM to extract information from provided passages."""

        if not passages:
            return None

        if not self.client:
            return self._fallback_extract(task, passages) if allow_fallback else None

        cfg = get_config()
        
        # 🔥 DIRECT TABLE PARSING for list/roster tasks (bypass LLM uncertainty)
        if self._is_list_task(task):
            direct_result = self._direct_table_extract(task, passages)
            if direct_result and direct_result.value and direct_result.confidence >= 0.7:
                print(f"[EXTRACTOR][DIRECT] Successfully extracted {task.variable_name} via table parsing (confidence={direct_result.confidence:.2f})")
                return direct_result
            else:
                print(f"[EXTRACTOR][DIRECT] Table parsing failed or low confidence, falling back to LLM")
        
        combined_text, provenance, raw_texts = self._format_passages(passages, mode)

        if not combined_text.strip():
            return None

        lang = self._detect_language(task.fact + " " + combined_text[:500])

        # Build structured extraction prompt
        plan_tasks_json = json.dumps([{
            "variable_name": task.variable_name,
            "fact": task.fact,
            "category": task.category
        }], ensure_ascii=False)

        # Extract time context
        time_rules = ""
        if ctx and hasattr(ctx, 'time_context') and ctx.time_context:
            tc = ctx.time_context
            if tc.window and tc.window[0]:
                target_date = tc.window[0].split('T')[0]
                if lang == "zh":
                    time_rules = f"\n\n时间规则：目标日期 = {target_date}。只接受明确标注此日期的数据，忽略「今天」、「当前」、「实时」标签。"
                else:
                    time_rules = f"\n\nTime Rule: Target date = {target_date}. Only accept data explicitly marked with this date. Ignore 'today', 'current', 'real-time' labels."

        if lang == "zh":
            system_msg = """你是“结构化提取专家（严格模式）”。只从 EVIDENCE 中抽取 PLAN_TASKS 需要的事实；绝不外推或引入常识。严格对齐、就近取证、时间可验证。

【范围/话题控制】
- 仅处理 PLAN_TASKS 中的变量；与任务无关的事实一概忽略。
- 若 EVIDENCE 含多实体，仅在"目标实体及其别名"就近范围内取值。

【提取模式选择 - 优先级规则】
**首先检查是否为"列表提取模式"（见下方专门规则）。如果是，跳过"三锚规则"，直接使用表格/列表提取逻辑。**
**只有非列表类任务才需要严格遵循"三锚规则"。**

【实体与时间的"三锚"规则】（仅适用于非列表类任务）
抽取任何"带时间的事实"（事件/奖项/任免/任期等）时，必须在同一来源文本的"同句/同行或±120字符"内，同时出现：
1) 目标实体的明确提及（姓名/机构/别名之一）；
2) 事件触发词（见下）或等价短语；
3) 可规约的日期表达（YYYY[-MM[-DD]] 或“YYYY–YYYY”区间等）。
三者任一缺失 → 该候选标注 `proximity_ok:false`；若无任何 `proximity_ok:true` 的候选 → `status:"unverified"`。

【事件触发词词表（部分）】
- 当选/当選/elected/reelected、任命/appointed、就任/assumed office/sworn in/inaugurated、代理/acting、续任/连任/reappointed、卸任/stepped down/retired/removed、任期开始/结束/term start/end、获授/授予/被评为/awarded/conferred、当选院士/fellow of…
- 对头衔/职位变动，区分：获任命 vs 就职（实际开始） vs 卸任；不可混用。

【日期与任期规约】
- 仅接受与实体+事件触发词相邻近的日期；忽略页面元数据（“更新于/发布于/最后修改”）和“as of/截至/目前/现任”类“状态时间”。
- 解析范围：精确日/月/年、季度（Q1–Q4→以季度首月规约）、中英文日期、中文数字日期、破折号区间（如 2015–2018、2015年1月–2017年3月）。
- “至今/在任/现任/present”→ 仅在明确存在“任期开始”且实体匹配时，输出 open-ended 结束，值用字符串区间（如 "2019-03–present"），precision 取开始端精度；否则 `unverified`。
- 不从“获奖年份 X”推断“任期开始/毕业年份”等任何未明示的时间。

【🔥 列表提取模式 - 优先规则（覆盖三锚规则）🔥】
**触发条件**：variable_name 包含 "list/roster/chiefs/heads/members/leaders/names"，或 category="aggregation" 且要求多个实体

**一旦触发列表提取模式，完全忽略"三锚规则"和"事件触发词"要求，改用以下简化规则：**

1. **[TABLE]...[/TABLE] 表格提取**：
   - 识别列头：姓名/Name、职位/Position/Title、年份/Year/Since/From/Appointed
   - 提取表格中**所有行**的数据：
     * **必需**：姓名 + 职位（同一行内）
     * **可选**：年份（如果表格列有年份就提取，没有就不提取）
     * ⚠️ **即使任务名是 "with_years"，也要提取没有年份的条目**
   - ⚠️ 不需要"事件触发词"（表格本身就是事实陈述）
   - ⚠️ 不需要验证"±120字符"规则
   - 格式化输出：
     * 有年份："职位1：姓名1（2022年7月）; 职位2：姓名2（2020年）; ..."
     * 无年份："职位1：姓名1; 职位2：姓名2; ..."
     * 混合情况："职位1：姓名1（2022年）; 职位2：姓名2; 职位3：姓名3（2023年）; ..."

2. **文本列表提取**（如"局长：XXX（2022年）"）：
   - 识别模式："职位[：:] 姓名 [可选：(年份)]"
   - 提取所有匹配的名字

3. **输出要求**：
   - ✅ 正确示例："公務員事務局長：楊何蓓茵; 政制及內地事務局長：曾國衞（2022年7月）; 文化體育及旅遊局長：羅淑佩; ..."
   - ❌ 错误示例："有15个局长"（禁止总结，必须列出所有名字）
   - ❌ 错误示例："包括多个决策局"（必须给出具体名单）

4. **Confidence 评分**（放宽标准）：
   - 找到 ≥3 个实体 → status:"ok", confidence=0.9
   - 找到 1-2 个 → status:"ok", confidence=0.7
   - 一个都没找到 → status:"unverified", confidence=0.6

**关键**：列表模式下，即使缺少年份或事件触发词，只要有姓名+职位，就应该提取并返回 status:"ok"。

【冲突与去歧义】
- 同一变量若不同来源/同来源不同片段给出**不相同值**且均满足“三锚”→ `status:"conflict"`；在 candidates 中列出全部，注明距离与触发词。
- 若值相同但精度不同，选择精度更高者为主值；保留另一个为候选。
- 源优先级（仅作并列时的最后决策因子）：官方/主办方/法规 > 学校/研究所/协会 > 一线媒体/数据库 > 聚合器/百科/个人博客。

【输出质量门槛（任一不满足则降级为 unverified）】
- 缺少明确实体锚点；或实体锚点与日期/事件不在同域近邻（±120字符）；
- 仅出现“被提名/拟任/传闻/预测/待审”；
- 仅出现“在任/截至某日仍任”却无开始时间且 PLAN_TASKS 要求开始/结束；
- 任期区间开始>结束或重叠/倒置；
- 院士/fellow 未给出完整授予机构名。

【集合/基数控制】
- PLAN_TASKS 可为每个变量标记 cardinality: "SINGLE" | "LIST"（如缺省，按 SINGLE 处理）。
- 当 cardinality="LIST"：返回按 items[] 组织的集合；每个 item 都必须满足“三锚近邻”（实体+事件触发词+日期）才计入。若无任何满足项 → status:"unverified"。
- 可尊重 PLAN_TASKS 的限制：max_items、year_range、type_filter（如 {"type":["award","education"]}），用于控制数量与范围。

【去重/归并与冲突】
- 对 LIST：先对候选按（规范化名称 + 组织/院系 + 主日期/起止区间）聚类去重；同簇内日期一致且精度更高者为主值，其余入 candidates。
- 同簇内若出现**不同值且均满足三锚** → 该簇标注 conflict:true，并在 notes 与 candidates 中展示；若存在 ≥1 个冲突簇 → 本变量 status:"conflict"；否则若 items 非空 → status:"ok"；否则 "unverified"。

【候选记录要求（用于审核）】
- 每个候选须给出：value、precision、source_id、url、proximity_ok、触发词（event_trigger）、实体命中（entity_mention）、日期原文（date_mention）、近邻距离（distance_chars）、引用片段（quote ≤ 200 字）。

严格遵循以下 I/O 模板。除 JSON 外不得输出任何文字。
"""
            prompt = f"""INPUT
PLAN_TASKS: {plan_tasks_json}
EVIDENCE: {combined_text}{time_rules}

OUTPUT JSON SCHEMA
{{
  "results": [
    {{
      "variable_name": "{task.variable_name}",
      "status": "ok" | "unverified" | "conflict",
      "value": "string|number|null",
      "precision": "day"|"month"|"year"|null,
      "source_ids": [1,2],
      "candidates": [
        {{
          "value":"...",
          "precision":"day|month|year|null",
          "source_id":1,
          "url":"...",
          "proximity_ok": true,
          "event_trigger":"appointed|elected|term start|award|...",
          "entity_mention":"原文中的实体提及",
          "date_mention":"原文中的日期表达",
          "distance_chars": 87,
          "quote":"（含实体/触发词/日期的原文片段 ≤200字）"
        }}
      ],
      "notes": "简短原因（如：实体/事件/日期三锚齐备；或为何 unverified/conflict）"
    }}
  ]
}}

仅返回JSON，不要解释。"""
        else:
            system_msg = """You are a "Structured Extractor – Strict Mode". Extract ONLY the facts required by PLAN_TASKS from EVIDENCE. No world knowledge, no guessing.

[Scope & Topic Control]
- Only handle variables listed in PLAN_TASKS; ignore everything else.
- If multiple entities appear, extract values ONLY within the local neighborhood of the target entity or its aliases.

[Extraction Mode Selection - Priority Rule]
**First check if this is "List Extraction Mode" (see dedicated rules below). If yes, SKIP the "Triple-Anchor Rule" and use table/list extraction logic directly.**
**Only non-list tasks need to strictly follow the "Triple-Anchor Rule".**

[Entity–Event–Date "Triple-Anchor" Rule] (only for non-list tasks)
For any time-stamped fact (events/awards/appointments/tenure), in the SAME source text and within the SAME sentence/line or ±120 characters, you must find:
1) an explicit mention of the target entity (name/alias/org),
2) an event trigger (see list below) or equivalent phrase,
3) a normalizable date expression (YYYY[-MM[-DD]] or a clear range).
If any anchor is missing → mark that candidate `proximity_ok:false`. If NO candidate has `proximity_ok:true` → `status:"unverified"`.

[Event Trigger Lexicon (partial)]
elected/reelected, appointed, assumed office/sworn in/inaugurated, acting, reappointed, stepped down/retired/removed, term start/term end, awarded/conferred, fellow of …

[Date & Tenure Normalization]
- Accept dates ONLY near the entity + event trigger; ignore page meta dates (“updated on/published on/last modified”) and status-time phrases (“as of/currently”).
- Parse day/month/year, quarters (Q1–Q4→map to first month), EN/CN dates, ranges (e.g., 2015–2018, Jan 2015–Mar 2017).
- “present/current” is allowed only if a clear term start exists for the entity; encode as "YYYY[-MM[-DD]–present" with precision from the start; otherwise `unverified`.
- Do NOT infer missing dates (e.g., do not derive start year from award year).

[🔥 List Extraction Mode - Priority Rule (Overrides Triple-Anchor) 🔥]
**Trigger**: variable_name contains "list/roster/chiefs/heads/members/leaders/names", OR category="aggregation" requiring multiple entities

**Once List Extraction Mode is triggered, COMPLETELY IGNORE "Triple-Anchor Rule" and "event trigger" requirements. Use these simplified rules instead:**

1. **[TABLE]...[/TABLE] Table Extraction**:
   - Identify headers: Name, Position/Title, Year/Since/From/Appointed
   - Extract **ALL rows** from table:
     * **Required**: name + position (in same row)
     * **Optional**: year (extract if column exists, skip if not available)
     * ⚠️ **Even if task name is "with_years", extract entries WITHOUT year too**
   - ⚠️ NO "event trigger" required (table itself is factual statement)
   - ⚠️ NO "±120 chars" proximity validation needed
   - Format output:
     * With year: "Position1: Name1 (July 2022); Position2: Name2 (2020); ..."
     * Without year: "Position1: Name1; Position2: Name2; ..."
     * Mixed: "Position1: Name1 (2022); Position2: Name2; Position3: Name3 (2023); ..."

2. **Text List Extraction** (e.g., "Chief: John Doe (2022)"):
   - Recognize pattern: "position[: ] name [optional: (year)]"
   - Extract all matching names

3. **Output Requirements**:
   - ✅ GOOD: "Secretary for the Civil Service: Ada Chung; Secretary for Constitutional and Mainland Affairs: Erick Tsang (July 2022); ..."
   - ❌ BAD: "There are 15 bureau chiefs" (no summarizing, must list all names)
   - ❌ BAD: "Multiple policy bureaus" (must provide specific roster)

4. **Confidence Scoring** (relaxed standards):
   - Found ≥3 entities → status:"ok", confidence=0.9
   - Found 1-2 entities → status:"ok", confidence=0.7
   - Found none → status:"unverified", confidence=0.6

**Critical**: In list mode, even without year or event trigger, if name+position exists, extract it and return status:"ok".

[Conflicts & Disambiguation]
- If TWO different values both satisfy the triple-anchor rule → `status:"conflict"` and list all in candidates with distance and trigger.
- If values match but precisions differ → choose the higher precision as main value; keep the other as candidate.
- Source priority (tie-breaker only): official/organizer/law > university/association > tier-1 media/databases > aggregators/wiki/blogs.

[Quality Gates → downgrade to unverified if ANY holds]
- Missing entity anchor; OR entity/date not near the event trigger (±120 chars);
- Nomination/rumor/prediction/pending only;
- “as of/currently” without a clear start date when start/end is required;
- Term ranges reversed/overlapping;
- “fellow/academician” without a full granting organization name.

[Candidate Record (for audit)]
Each candidate must include: value, precision, source_id, url, proximity_ok, event_trigger, entity_mention, date_mention, distance_chars, and a short quote (≤200 chars) containing the three anchors.

Return STRICT JSON only. No extra text.
"""
            prompt = f"""INPUT
PLAN_TASKS: {plan_tasks_json}
EVIDENCE: {combined_text}{time_rules}

OUTPUT JSON SCHEMA
{{
  "results": [
    {{
      "variable_name": "{task.variable_name}",
      "status": "ok" | "unverified" | "conflict",
      "value": "string|number|null",
      "precision": "day"|"month"|"year"|null,
      "source_ids": [1,2],
      "candidates": [
        {{
          "value":"...",
          "precision":"day|month|year|null",
          "source_id":1,
          "url":"...",
          "proximity_ok": true,
          "event_trigger":"appointed|elected|term start|award|...",
          "entity_mention":"exact mention in text",
          "date_mention":"raw date string",
          "distance_chars": 87,
          "quote":"(≤200 chars snippet containing entity+trigger+date)"
        }}
      ],
      "notes": "short reason (e.g., triple-anchor satisfied; or why unverified/conflict)"
    }}
  ]
}}

Return JSON only, no explanation."""

        model_name = model_name or self.model_name
        max_tokens = max_tokens or cfg.get('models.extractor.max_tokens', 4000)

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
        except Exception as api_error:
            print(f"[EXTRACTOR][WARN] response_format not supported ({api_error}), retrying without it")
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

        message = response.choices[0].message
        result_text = (message.content or "").strip()
        if not result_text:
            print(f"[EXTRACTOR][ERROR] Empty LLM response for {task.variable_name}")
            return self._fallback_extract(task, passages) if allow_fallback else None

        try:
            result_data = json.loads(result_text)
        except json.JSONDecodeError as err:
            print(f"[EXTRACTOR][ERROR] JSON decode failed for {task.variable_name}: {err}")
            print(f"[EXTRACTOR][DEBUG] Raw response: {result_text[:200]}")
            return self._fallback_extract(task, passages) if allow_fallback else None

        # Parse new structured format
        results = result_data.get("results", [])
        if not results:
            print(f"[EXTRACTOR][WARN] No results in structured response for {task.variable_name}")
            return self._fallback_extract(task, passages) if allow_fallback else None
        
        result = results[0]  # Get first result (single task extraction)
        status = result.get("status", "ok")
        value = result.get("value")
        notes = result.get("notes", "")
        
        # Calculate confidence based on status and candidates
        if status == "ok":
            confidence = 0.95
        elif status == "unverified":
            confidence = 0.60
        elif status == "conflict":
            confidence = 0.40
        else:
            confidence = 0.50
        
        # If multiple candidates, reduce confidence slightly
        candidates = result.get("candidates", [])
        if len(candidates) > 1:
            confidence *= 0.9
        
        # 🔥 FALLBACK: If extraction failed (None or low confidence), use raw material
        if (not value or confidence < 0.7) and raw_texts:
            print(f"[EXTRACTOR][FALLBACK] Extraction failed for {task.variable_name}, using raw material")
            # Combine relevant raw texts (limited to avoid huge responses)
            raw_material = self._extract_relevant_raw_material(raw_texts, task)
            if raw_material and len(raw_material) > 50:  # At least some meaningful content
                value = raw_material
                confidence = 0.65  # Low but not too low
                status = "raw_fallback"
                notes = f"LLM提取失败，返回原始材料段落。原因：{notes}"
                print(f"[EXTRACTOR][FALLBACK] Using {len(raw_material)} chars of raw material")
        
        # Map source_ids to provenance URLs
        source_ids = result.get("source_ids", [])
        selected_provenance = [provenance[i-1] for i in source_ids if 1 <= i <= len(provenance)]
        if not selected_provenance:
            selected_provenance = provenance  # Fallback to all

        if cfg.is_decision_logging_enabled('ir_rag'):
            print(
                f"[EXTRACTOR][DEBUG] {stage_label} {task.variable_name}: value={str(value)[:50] if value else 'None'}... "
                f"confidence={confidence:.2f} status={status} notes={notes[:50]}..."
            )

        return ExtractedVariable(
            variable_name=task.variable_name,
            value=value,
            confidence=confidence,
            provenance=selected_provenance,
            extraction_method=f"{stage_label}-structured" if status != "raw_fallback" else "raw-fallback",
            raw_passages=raw_texts
        )


    def _fallback_extract(self, task: PlanTask, passages: List[ContentPassage]) -> ExtractedVariable:
        """Fallback extraction using simple heuristics"""
        if not passages:
            return ExtractedVariable(
                variable_name=task.variable_name,
                value=None,
                confidence=0.0,
                extraction_method="fallback"
            )

        # Use the highest-scored passage as the answer
        best_passage = passages[0]

        # Simple extraction based on task category
        if task.category == "biography":
            value = self._extract_biographical_info(best_passage.text)
        elif task.category == "fact_verification":
            value = self._extract_factual_claim(best_passage.text)
        else:
            # Generic extraction - use first sentence or paragraph
            sentences = best_passage.text.split('.')
            value = sentences[0].strip() if sentences else best_passage.text[:200]

        return ExtractedVariable(
            variable_name=task.variable_name,
            value=value,
            confidence=0.6,
            provenance=[(p.metadata or {}).get('provenance', p.source_url) for p in passages],
            extraction_method="fallback",
            raw_passages=[(p.raw_text or p.text) for p in passages]
        )
    
    def _extract_biographical_info(self, text: str) -> str:
        """Extract biographical information"""
        # Look for patterns like "born in", "graduated from", etc.
        bio_patterns = [
            r'born (?:in|on) ([^.]+)',
            r'graduated from ([^.]+)',
            r'worked at ([^.]+)',
            r'known for ([^.]+)'
        ]
        
        for pattern in bio_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback to first sentence
        sentences = text.split('.')
        return sentences[0].strip() if sentences else text[:200]
    
    def _extract_factual_claim(self, text: str) -> str:
        """Extract factual claims"""
        # Look for definitive statements
        fact_patterns = [
            r'(was founded in \d{4})',
            r'(established in \d{4})',
            r'(created in \d{4})',
            r'(\d{4}[^.]*founded)'
        ]
        
        for pattern in fact_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback to first sentence
        sentences = text.split('.')
        return sentences[0].strip() if sentences else text[:200]



# ============================================================================
# Test/Demo Code
# ============================================================================

if __name__ == "__main__":
    print("[TEST][Extractor] InformationExtractor module loaded successfully")
    print(f"[TEST][Extractor] Module contains {len(InformationExtractor.__dict__)} methods")
    print("=" * 80)

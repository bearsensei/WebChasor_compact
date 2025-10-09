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
        Run Stage 1 extraction concurrently for all tasks.
        Returns list of (task, passages, stage1_result) tuples.
        """
        if not tasks:
            return []

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
        solved = sum(1 for _, result in stage2_outputs if result and result.confidence >= 0.7)
        print(
            f"[EXTRACTOR][Stage2] Completed {len(stage2_outputs)} tasks in {elapsed:.2f}s "
            f"(confidence ≥0.7: {solved}, concurrent)"
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

        # Deduplicate by text to avoid redundant context
        seen_text = set()
        unique_passages = []
        for passage in selected:
            text_key = passage.text.strip()[:200]
            if text_key in seen_text:
                continue
            seen_text.add(text_key)
            unique_passages.append(passage)

        return unique_passages

    def _format_passages(self, passages: List[ContentPassage], mode: str) -> tuple:
        """Create human-readable blocks and provenance for prompts."""
        blocks = []
        provenance = []
        raw_texts = []

        for idx, passage in enumerate(passages, 1):
            meta = passage.metadata or {}
            label = "Snippet" if passage.source_url == "SERP_SNIPPET" else "Passage"
            title = meta.get('title') or meta.get('page_title') or ""
            provenance_id = meta.get('provenance', passage.source_url)
            snippet_date = meta.get('snippet_date')
            if mode == "snippet" and meta.get('snippet'):
                body_text = meta['snippet']
            else:
                body_text = passage.text

            raw_texts.append(body_text)
            provenance.append(provenance_id)

            header_parts = [f"{label} {idx}"]
            if title:
                header_parts.append(f"Title: {title}")
            header_parts.append(f"Source: {provenance_id}")
            if snippet_date:
                header_parts.append(f"Date: {snippet_date}")

            block = "\n".join(header_parts + [body_text])
            blocks.append(block)

        combined_text = "\n\n".join(blocks)
        return combined_text, provenance, raw_texts

    def _cache_key(self, task: PlanTask, passages: List[ContentPassage], stage_label: str) -> str:
        digest = hashlib.sha1()
        digest.update(task.variable_name.encode('utf-8'))
        digest.update(stage_label.encode('utf-8'))
        for passage in passages:
            digest.update(passage.text[:200].encode('utf-8', errors='ignore'))
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
        combined_text, provenance, raw_texts = self._format_passages(passages, mode)

        if not combined_text.strip():
            return None

        core_entities = self._extract_core_entities(task.fact)
        entities_str = ", ".join(core_entities) if core_entities else "N/A"
        lang = self._detect_language(task.fact + " " + combined_text[:500])

        # Extract time context
        time_info = ""
        if ctx and hasattr(ctx, 'time_context') and ctx.time_context:
            tc = ctx.time_context
            if tc.window and tc.window[0]:
                target_date = tc.window[0].split('T')[0]  # Extract YYYY-MM-DD
                if lang == "zh":
                    time_info = f"\n\n🔴 严格时间约束：目标日期为 {target_date}。\n- 只能提取标注为 {target_date} 或「昨天」、「昨日」的数据\n- 如果数据标注为「今天」、「当前」、「盘前」、「实时」，则不是 {target_date} 的数据，必须忽略\n- 如果看到多个价格数据，必须找出明确属于 {target_date} 的那个\n- 如果无法确定日期，返回 null 和低置信度"
                else:
                    time_info = f"\n\n🔴 STRICT TIME CONSTRAINT: Target date is {target_date}.\n- Only extract data explicitly marked as {target_date} or 'yesterday'\n- If data is marked as 'today', 'current', 'pre-market', or 'real-time', it is NOT from {target_date} and must be ignored\n- If multiple prices exist, find the one explicitly for {target_date}\n- If date cannot be confirmed, return null with low confidence"

        context_label = "这些搜索摘要" if lang == "zh" and mode == "snippet" else (
            "这些文本段落" if lang == "zh" else "the search snippets" if mode == "snippet" else "the text passages"
        )

        if lang == "zh":
            priority_note = "\n\n数据源优先级：\n1. 标注「成交」/「收盘」且来自券商数据的Answer Box\n2. 搜索摘要中明确标注日期的数据\n3. 网页正文\n4. 历史数据表格（需确认日期匹配）" if mode == "snippet" else "\n\n数据验证：提取的数据必须来自文本中明确提到的段落，并在 source_quote 中引用原文。"
            prompt = (
                f"请从{context_label}中提取以下信息，并确保仅依据提供的内容。{time_info}{priority_note}\n\n"
                f"问题: {task.fact}\n变量名: {task.variable_name}\n类别: {task.category}\n核心实体: {entities_str}\n\n"
                f"{context_label}:\n{combined_text}\n\n"
                '输出 JSON：{"value": str|null, "confidence": 0-1, "reasoning": str, "source_quote": str}。\n'
                '注意：reasoning 必须说明该数据来自哪个日期、哪个来源；source_quote 必须包含原文的日期标注。'
            )
            system_msg = "你是一个专业的信息提取专家，擅长时间敏感的数据提取。请仅返回有效 JSON，且完全依赖提供的文本。规则：1) 当看到「今天」、「当前」、「实时」等词时，这些数据不是目标日期的数据；2) 只提取明确标注了目标日期的数据；3) 如果有多个价格，必须找出日期匹配的那个；4) 不确定时返回 null。"
        else:
            priority_note = "\n\nData Source Priority:\n1. Answer Box with 'closing'/'成交' label from broker data\n2. Search snippets with explicit date markers\n3. Web page content\n4. Historical tables (must confirm date match)" if mode == "snippet" else "\n\nData Validation: Extracted data must come from explicitly mentioned passages, with source_quote citing the original text."
            prompt = (
                f"Extract the requested fact strictly from {context_label}. No guessing.{time_info}{priority_note}\n\n"
                f"QUESTION: {task.fact}\nVARIABLE: {task.variable_name}\nCATEGORY: {task.category}\nCORE ENTITIES: {entities_str}\n\n"
                f"{context_label.upper()}:\n{combined_text}\n\n"
                'Return valid JSON: {"value": str|null, "confidence": 0-1, "reasoning": str, "source_quote": str}.\n'
                'Note: reasoning must specify which date and source; source_quote must include date markers from the original text.'
            )
            system_msg = "You are an expert in time-sensitive data extraction. Respond with JSON only using the supplied context. Rules: 1) When you see 'today', 'current', 'real-time', that data is NOT from the target date; 2) Only extract data explicitly marked with the target date; 3) If multiple prices exist, find the one matching the date; 4) Return null if uncertain."

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

        if cfg.is_decision_logging_enabled('ir_rag'):
            print(
                f"[EXTRACTOR][DEBUG] {stage_label} {task.variable_name}: value={result_data.get('value')}"
                f" confidence={result_data.get('confidence')}"
            )

        return ExtractedVariable(
            variable_name=task.variable_name,
            value=result_data.get("value"),
            confidence=float(result_data.get("confidence", 0.0)),
            provenance=provenance,
            extraction_method=stage_label,
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
            raw_passages=[p.text for p in passages]
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

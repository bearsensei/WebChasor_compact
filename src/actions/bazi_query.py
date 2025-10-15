"""
Bazi Query Action for WebChasor
Handles Chinese fortune telling / Bazi chart queries
"""
import re
import asyncio
from typing import Optional
from dataclasses import dataclass

from artifacts import Action, Context, Artifact
from utils.bazi import compute_bazi
from config_manager import get_config


@dataclass
class BaziIntent:
    """Parsed Bazi query intent"""
    year: int
    month: int
    day: int
    hour: int = 12
    minute: int = 0
    tz_name: str = "Asia/Shanghai"
    longitude: Optional[float] = None
    use_true_solar_time: bool = True


class BAZI_QUERY(Action):
    """Action for handling Bazi (八字) fortune telling queries"""
    
    name = "BAZI_QUERY"
    requires_tools = []
    max_time_s = 10
    max_tokens_in = 1000
    max_tokens_out = 2000
    
    def __init__(self):
        """Initialize Bazi query handler"""
        print(f"[BAZI_QUERY][INIT] Bazi calculator initialized")
        
        # Load config for enhancement
        cfg = get_config()
        self.enable_enhancement = cfg.get('bazi_query.enable_enhancement', True)
        self.enhancement_temperature = cfg.get('bazi_query.enhancement_temperature', 0.5)
        self.model_name = cfg.get('bazi_query.model_name', 'gpt-oss-20b')
        self.provider = cfg.get('bazi_query.provider', 'openai')
        
        # Initialize synthesizer if enhancement is enabled
        self.synthesizer = None
        if self.enable_enhancement:
            try:
                from synthesizer import Synthesizer
                self.synthesizer = Synthesizer()
                print(f"[BAZI_QUERY][INIT] Synthesizer initialized for enhancement (model={self.model_name})")
            except Exception as e:
                print(f"[BAZI_QUERY][WARN] Failed to initialize synthesizer: {e}")
                self.enable_enhancement = False
    
    async def run(self, ctx: Context, toolset) -> Artifact:
        """Execute Bazi query"""
        print(f"[BAZI_QUERY][START] query='{ctx.query}'")
        
        try:
            intent = self._parse_bazi_intent(ctx.query)
            print(f"[BAZI_QUERY][INTENT] date={intent.year}-{intent.month}-{intent.day} {intent.hour}:{intent.minute}")
            
            result = await self._compute_bazi_chart(intent)
            
            if self.enable_enhancement and self.synthesizer:
                result = await self._enhance_response(result, ctx)
            
            print(f"[BAZI_QUERY][COMPLETE]")
            return result
            
        except Exception as e:
            print(f"[BAZI_QUERY][ERROR] {e}")
            import traceback
            traceback.print_exc()
            return Artifact(
                kind="text",
                content=f"抱歉，八字排盘计算出错：{str(e)}",
                meta={"error": str(e)}
            )
    
    def _parse_bazi_intent(self, query: str) -> BaziIntent:
        """Parse query to extract birth date/time"""
        # Pattern 1: YYYY年MM月DD日HH点
        pattern1 = r'(\d{4})年(\d{1,2})月(\d{1,2})日.*?(\d{1,2})[点时]'
        match = re.search(pattern1, query)
        if match:
            y, m, d, h = map(int, match.groups())
            return BaziIntent(year=y, month=m, day=d, hour=h, minute=0)
        
        # Pattern 2: YYYY-MM-DD HH:MM
        pattern2 = r'(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2})'
        match = re.search(pattern2, query)
        if match:
            y, m, d, h, mi = map(int, match.groups())
            return BaziIntent(year=y, month=m, day=d, hour=h, minute=mi)
        
        # Pattern 3: Extract any date
        pattern3 = r'(\d{4}).*?(\d{1,2}).*?月.*?(\d{1,2}).*?日'
        match = re.search(pattern3, query)
        if match:
            y, m, d = map(int, match.groups())
            hour_match = re.search(r'(\d{1,2})[点时:：]', query)
            h = int(hour_match.group(1)) if hour_match else 12
            return BaziIntent(year=y, month=m, day=d, hour=h, minute=0)
        
        raise ValueError(f"无法从查询中提取生辰信息：{query}")
    
    async def _compute_bazi_chart(self, intent: BaziIntent) -> Artifact:
        """Compute Bazi chart"""
        print(f"[BAZI_QUERY][COMPUTE] Computing chart...")
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                compute_bazi,
                intent.year,
                intent.month,
                intent.day,
                intent.hour,
                intent.minute,
                intent.tz_name,
                intent.longitude,
                intent.use_true_solar_time
            )
            
            content = self._format_bazi_result(result, intent)
            
            return Artifact(
                kind="text",
                content=content,
                meta={
                    "bazi_type": "chart",
                    "birth_date": f"{intent.year}-{intent.month:02d}-{intent.day:02d}",
                    "birth_time": f"{intent.hour:02d}:{intent.minute:02d}",
                    "pillars": result['pillars']
                }
            )
        except Exception as e:
            print(f"[BAZI_QUERY][ERROR] Computation failed: {e}")
            return Artifact(
                kind="text",
                content=f"八字计算出错：{str(e)}",
                meta={"error": str(e)}
            )
    
    def _format_bazi_result(self, result: dict, intent: BaziIntent) -> str:
        """Format Bazi computation result"""
        pillars = result['pillars']
        
        lines = [
            f"八字排盘结果",
            f"出生时间：{intent.year}年{intent.month}月{intent.day}日 {intent.hour}时{intent.minute}分",
            f"",
            f"四柱八字：",
            f"  年柱：{pillars['year']}",
            f"  月柱：{pillars['month']}",
            f"  日柱：{pillars['day']}",
            f"  时柱：{pillars['hour']}",
            f"",
            f"完整八字：{pillars['year']} {pillars['month']} {pillars['day']} {pillars['hour']}"
        ]
        
        return "\n".join(lines)
    
    async def _enhance_response(self, result: Artifact, ctx: Context) -> Artifact:
        """Enhance response using LLM"""
        print(f"[BAZI_QUERY][ENHANCE] Starting enhancement...")
        
        try:
            from prompt import build_bazi_query_instruction, get_length_hint
            
            cfg = get_config()
            length_config = cfg.get_response_length_config("BAZI_QUERY")
            max_tokens = length_config.get('max_tokens', 800)
            temperature = length_config.get('temperature', 0.5)
            
            instruction_hint = build_bazi_query_instruction(ctx.query)
            instruction_hint += get_length_hint(max_tokens)
            
            materials = f"""# 八字排盘基础信息

{result.content}

# 五行对应参考
天干：甲乙（木）、丙丁（火）、戊己（土）、庚辛（金）、壬癸（水）
地支：寅卯（木）、巳午（火）、辰戌丑未（土）、申酉（金）、亥子（水）
"""
            
            enhanced_content = await self.synthesizer.generate(
                category="BAZI_QUERY",
                style_key="auto",
                constraints={
                    "language": "zh",
                    "tone": "professional, respectful, concise",
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "instruction_hint": instruction_hint,
                    "model_name": self.model_name,
                    "provider": self.provider
                },
                materials=materials,
                task_scaffold=None
            )
            
            print(f"[BAZI_QUERY][ENHANCE] Completed (max_tokens={max_tokens})")
            
            return Artifact(
                kind=result.kind,
                content=enhanced_content,
                meta={
                    **result.meta,
                    "enhanced": True,
                    "max_tokens": max_tokens
                }
            )
            
        except Exception as e:
            print(f"[BAZI_QUERY][ENHANCE][ERROR] Enhancement failed: {e}")
            return result


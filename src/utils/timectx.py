# timectx.py
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re


class TimeIntent:
    LATEST = "latest"     # 近7天（默认）
    TREND  = "trend"      # 近30天/本年
    HISTORIC = "historic" # 过去固定年段
    STATIC = "static"     # 不强调时效

@dataclass
class TimeContext:
    now_iso: str
    tz: str
    intent: str                 # "latest" | "trend" | "historic" | "static"
    window: tuple[str,str]      # (start_iso, end_iso)
    granularity: str            # "hour" | "day" | "week" | "month" | "year"
    explicit_dates: list[str]   # absolute dates found in query (ISO)
    display_cutoff: str         # what to print as "as of ..."

# src/utils/timectx.py
def parse_time_intent(query: str, tz: str="Asia/Hong_Kong") -> TimeContext:
    now = datetime.now(timezone.utc).astimezone().replace(microsecond=0)
    ql = query.lower()
    
    # 初始化默认值
    start = None
    end = now
    intent = "static"
    gran = "none"
    
    # 扩展的时间模式匹配
    if any(k in ql for k in ["latest", "today", "now", "最近", "最新", "当前"]):
        intent="latest"; gran="day"; start=now - timedelta(days=7)
    elif any(k in ql for k in ["past 30 days","过去30天","last month","上个月"]):
        intent="trend"; gran="day"; start=now - timedelta(days=30)
    elif any(k in ql for k in ["this year","今年","ytd","本年度"]):
        intent="latest"; gran="month"; start=now.replace(month=1, day=1, hour=0, minute=0, second=0)
    elif any(k in ql for k in ["yesterday","昨天"]):
        intent="historic"; gran="day"
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        end = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59)
    elif any(k in ql for k in ["last week","上周"]):
        intent="historic"; gran="week"
        start = now - timedelta(days=7)
        end = now
    elif any(k in ql for k in ["q1", "q2", "q3", "q4", "第一季度", "第二季度"]):
        # 处理季度查询
        intent="trend"; gran="month"
        if "q1" in ql or "第一季度" in ql:
            start=now.replace(month=1, day=1, hour=0, minute=0, second=0)
            end=now.replace(month=3, day=31, hour=23, minute=59, second=59)
        # ... 其他季度
    
    window=(start.isoformat() if start else "", end.isoformat())
    
    return TimeContext(
        now_iso=now.isoformat(),
        tz=tz,
        intent=intent,
        window=window,
        granularity=gran,
        explicit_dates=[],
        display_cutoff=now.date().isoformat()
    )
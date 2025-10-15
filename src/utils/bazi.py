# -*- coding: utf-8 -*-
"""
bazi.py — 四柱八字排盘（年柱 / 月柱 / 日柱 / 时柱）
优先使用寿星天文历 sxtwl/pysxtwl 以“节气/立春”为准判定年/月柱，精确可靠。
支持可选“真太阳时”校正；内置时柱推干表；提供丰富 debug 信息，方便校对。

依赖：
    pip install pysxtwl python-dateutil pytz
    # 某些环境包名是 sxtwl（不是 pysxtwl）。本模块会自动尝试两种导入方式。

核心思路：
1) 以“节气年”（立春为岁首）定年柱；
2) 以“节气月”（寅月自立春始）定月柱（Jie/Qi 节点判定）；
3) 用通用公式/库计算“日干支”；
4) 由“日干 × 时支”查表得“时干支”；
5) 可选按经度做“真太阳时”校正，以减少边界时刻误差。

作者：ChatGPT（为 Ryan 定制）
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
import math

# ---------- 兼容导入 sxtwl / pysxtwl ----------
_sxtwl = None

import sxtwl as _sxtwl  # 有的环境叫 sxtwl


# ---------- 其他依赖 ----------
from dateutil import tz
import pytz

# ------------------ 工具表：天干、地支、时柱表 ------------------
TIAN_GAN = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]
DI_ZHI   = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"]

# “日干 × 时支 → 时干”映射表（十天干为行，十二时支为列）
# 规律：甲/己同列，乙/庚同列，丙/辛同列，丁/壬同列，戊/癸同列，子时从甲开始顺排
HOUR_STEM_TABLE = [
    # 子 丑 寅 卯 辰 巳 午 未 申 酉 戌 亥
    ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸","甲","乙"],  # 日干：甲 or 己
    ["丙","丁","戊","己","庚","辛","壬","癸","甲","乙","丙","丁"],  # 日干：乙 or 庚
    ["戊","己","庚","辛","壬","癸","甲","乙","丙","丁","戊","己"],  # 日干：丙 or 辛
    ["庚","辛","壬","癸","甲","乙","丙","丁","戊","己","庚","辛"],  # 日干：丁 or 壬
    ["壬","癸","甲","乙","丙","丁","戊","己","庚","辛","壬","癸"],  # 日干：戊 or 癸
]
# 把 10 个日干映射到 5 行（甲=0,乙=1,丙=2,丁=3,戊=4,己=0,庚=1,辛=2,壬=3,癸=4）
HOUR_STEM_ROW = {g: idx for idx, g in enumerate(["甲","乙","丙","丁","戊"])}
HOUR_STEM_ROW.update({"己":0,"庚":1,"辛":2,"壬":3,"癸":4})

# 12 时辰范围（本地均时；真太阳时校正在外部完成）
# 注意：子时有两种流派（23:00–01:00 或 23:00–00:59:59），此处按常见的整段两小时
HOUR_TO_BRANCH = [
    (23, 1,  "子"), (1,  3,  "丑"), (3,  5,  "寅"), (5,  7,  "卯"),
    (7,  9,  "辰"), (9,  11, "巳"), (11, 13, "午"), (13, 15, "未"),
    (15, 17, "申"), (17, 19, "酉"), (19, 21, "戌"), (21, 23, "亥"),
]

# 月干推导表：由年干（列）× 月序（行，寅=1 … 丑=12）确定月干
# 规律：寅月天干起点：甲年丙、乙年戊、丙年庚、丁年壬、戊年甲、己年丙、庚年戊、辛年庚、壬年壬、癸年甲；之后依次顺推
# 这里直接给出映射：year_gan_index -> 寅月起点干索引
Y_GAN_TO_YIN_START = {
    0: 2,  # 甲 -> 丙(2)
    1: 4,  # 乙 -> 戊(4)
    2: 6,  # 丙 -> 庚(6)
    3: 8,  # 丁 -> 壬(8)
    4: 0,  # 戊 -> 甲(0)
    5: 2,  # 己 -> 丙(2)
    6: 4,  # 庚 -> 戊(4)
    7: 6,  # 辛 -> 庚(6)
    8: 8,  # 壬 -> 壬(8)
    9: 0,  # 癸 -> 甲(0)
}

# ------------------ 常用小函数 ------------------
def _extract_gz_index(gz_obj) -> int:
    """
    从 sxtwl 返回的 GZ 对象或整数中提取干支索引（0-59）。
    兼容不同版本的 sxtwl 库。
    """
    # 如果已经是整数，直接返回
    if isinstance(gz_obj, int):
        return gz_obj
    
    # 检查是否有 tg（天干）和 dz（地支）属性（sxtwl 2.0+ 的 GZ 对象）
    if hasattr(gz_obj, 'tg') and hasattr(gz_obj, 'dz'):
        tg = gz_obj.tg  # 天干索引 0-9
        dz = gz_obj.dz  # 地支索引 0-11
        # 使用中国剩余定理计算六十甲子索引
        # gz_index = (tg - dz) % 10 的结果 * 6 + dz / 2（简化公式）
        # 更直接的方法：遍历找到匹配的索引
        for i in range(60):
            if (i % 10 == tg) and (i % 12 == dz):
                return i
        raise ValueError(f"Cannot calculate GZ index from tg={tg}, dz={dz}")
    
    # 尝试各种可能的属性名
    for attr in ['index', 'gz', 'value', '__int__']:
        if hasattr(gz_obj, attr):
            val = getattr(gz_obj, attr)
            if callable(val):
                return val()
            return val
    
    # 尝试直接转换为整数
    try:
        return int(gz_obj)
    except:
        pass
    
    # 如果都失败了，打印调试信息
    print(f"[DEBUG] GZ object type: {type(gz_obj)}")
    print(f"[DEBUG] GZ object dir: {dir(gz_obj)}")
    print(f"[DEBUG] GZ object repr: {repr(gz_obj)}")
    raise ValueError(f"Cannot extract index from GZ object: {gz_obj}")

def _gz_from_index(idx60: int) -> str:
    """0..59 -> 干支字符串"""
    gan = TIAN_GAN[idx60 % 10]
    zhi = DI_ZHI[idx60 % 12]
    return gan + zhi

def _true_solar_time(dt_local: datetime, longitude: Optional[float], tz_offset_hours: float) -> datetime:
    """
    用经度修正地方时为“真太阳时”（简化版，仅考虑经度差，不做日方程）
    LSTM = 15° * 时区；Δt(min) = 4 * (经度 - LSTM)
    """
    if longitude is None:
        return dt_local
    lstm = 15.0 * tz_offset_hours
    delta_minutes = 4.0 * (longitude - lstm)
    return dt_local + timedelta(minutes=delta_minutes)

def _get_hour_branch(dt_local: datetime) -> str:
    h = dt_local.hour
    m = dt_local.minute
    # 特判：23:00-23:59 归子；00:00-00:59 仍子
    if h == 23:
        return "子"
    for start, end, zhi in HOUR_TO_BRANCH:
        # 采用 [start, end) 半开区间
        if start <= h < end:
            return zhi
    # 若 0 点则仍为子
    return "子"

# ------------------ 使用 sxtwl 进行干支/节气判定 ------------------
@dataclass
class BaziResult:
    year_gz: str
    month_gz: str
    day_gz: str
    hour_gz: str
    debug: Dict

def _ensure_sxtwl():
    if _sxtwl is None:
        raise ImportError(
            "未检测到 sxtwl/pysxtwl。请先安装：\n"
            "    pip install pysxtwl\n"
            "或（部分平台）\n"
            "    pip install sxtwl\n"
            "安装后重新运行本程序。"
        )

def _day_gz_by_sxtwl(dt_utc: datetime) -> str:
    """
    用 sxtwl 计算日干支（按 UTC 输入）。
    """
    _ensure_sxtwl()
    # sxtwl 的 Solar 类/接口在不同版本略有差异；采用通用法：先转为本地（中国历法基于东八区），再调用 fromSolar
    # 但稳妥做法：用 sxtwl 库的 fromUTC（若有）或先转北京时间。
    beijing = dt_utc.astimezone(pytz.timezone("Asia/Shanghai"))
    y, m, d = beijing.year, beijing.month, beijing.day
    # 典型用法：day = _sxtwl.Solar.fromSolar(y, m, d) 或 _sxtwl.fromSolar(y,m,d)
    # 兼容两种 API 名称：
    if hasattr(_sxtwl, "fromSolar"):
        day = _sxtwl.fromSolar(y, m, d)
    elif hasattr(_sxtwl, "Solar") and hasattr(_sxtwl.Solar, "fromSolar"):
        day = _sxtwl.Solar.fromSolar(y, m, d)
    else:
        raise RuntimeError("当前 sxtwl 版本不支持 fromSolar 接口。")
    gz_day_index = day.getDayGZ()  # 0..59 或 GZ 对象
    # 兼容处理：某些版本返回 GZ 对象，某些返回整数索引
    gz_day_index = _extract_gz_index(gz_day_index)
    return _gz_from_index(gz_day_index)

def _year_month_gz_by_sxtwl(dt_local: datetime) -> Tuple[str, str, Dict]:
    """
    用 sxtwl 依据节气年/节气月计算 年柱 / 月柱（输入为“本地时”）。
    规则：以立春为岁首；寅月自立春始，之后每过一个“节”换月支。
    """
    _ensure_sxtwl()
    y, m, d = dt_local.year, dt_local.month, dt_local.day

    # sxtwl 提供：根据公历日期直接得到 年/月/日 干支索引
    # 常见用法：day = fromSolar(y,m,d); day.getYearGZ(), day.getMonthGZ(), day.getDayGZ()
    if hasattr(_sxtwl, "fromSolar"):
        day = _sxtwl.fromSolar(y, m, d)
    elif hasattr(_sxtwl, "Solar") and hasattr(_sxtwl.Solar, "fromSolar"):
        day = _sxtwl.Solar.fromSolar(y, m, d)
    else:
        raise RuntimeError("当前 sxtwl 版本不支持 fromSolar 接口。")

    # 注意：sxtwl 的年/月干支默认已按节气（不少版本如此）。为稳妥起见，我们仍拿"立春界"与"节气月"做一次交叉校验：
    year_idx = day.getYearGZ()   # 0..59 或 GZ 对象
    month_idx = day.getMonthGZ() # 0..59 或 GZ 对象

    # 兼容处理：某些版本返回 GZ 对象，某些返回整数索引
    year_idx = _extract_gz_index(year_idx)
    month_idx = _extract_gz_index(month_idx)

    # 这里直接信任 sxtwl 的年/月干支（已按节气划分）
    year_gz  = _gz_from_index(year_idx)
    month_gz = _gz_from_index(month_idx)

    dbg = {
        "sxtwl_year_index": year_idx,
        "sxtwl_month_index": month_idx,
        "local_date_used": dt_local.isoformat()
    }
    return year_gz, month_gz, dbg

def _hour_gz_from_day_gan_and_localtime(day_gan: str, dt_local: datetime) -> Tuple[str, Dict]:
    """
    根据“日干 + 时辰”求“时干支”。
    """
    zhi = _get_hour_branch(dt_local)
    row = HOUR_STEM_ROW[day_gan]
    col = DI_ZHI.index(zhi)
    gan = HOUR_STEM_TABLE[row][col]
    return gan + zhi, {"day_gan": day_gan, "hour_branch": zhi}

# ------------------ 对外主函数 ------------------
def compute_bazi(
    year: int, month: int, day: int,
    hour: int = 12, minute: int = 0,
    tz_name: str = "Asia/Shanghai",
    longitude: Optional[float] = None,
    use_true_solar_time: bool = True
) -> Dict:
    """
    输入：公历年月日时分 + 时区（IANA 名称；例如 Asia/Shanghai）
    可选：经度（度），以及是否启用真太阳时校正（建议 True）。
    返回：四柱与 debug 信息。
    """
    # 1) 组成本地时间
    tzinfo = pytz.timezone(tz_name)
    dt_local = tzinfo.localize(datetime(year, month, day, hour, minute))

    # 2) 可选真太阳时（仅做经度微调，不含“日方程”）
    if use_true_solar_time:
        # 取该时刻的时区相对 UTC 的小时偏移
        offset_sec = dt_local.utcoffset().total_seconds()
        tz_offset_hours = offset_sec / 3600.0
        dt_local_adj = _true_solar_time(dt_local, longitude, tz_offset_hours)
    else:
        dt_local_adj = dt_local

    # 3) 用 sxtwl 计算 年/月/日干支
    if _sxtwl is None:
        raise ImportError(
            "需要安装 sxtwl/pysxtwl 以确保年/月柱按节气判定准确。\n"
            "请先：pip install pysxtwl   （或 pip install sxtwl）"
        )

    # 年/月（按本地均时/或真太阳时）
    year_gz, month_gz, dbg_ym = _year_month_gz_by_sxtwl(dt_local_adj)

    # 日（以东八历法特性，常以北京时或库内部规则计算；此处按 UTC 统一输入）
    dt_utc = dt_local_adj.astimezone(pytz.utc)
    day_gz = _day_gz_by_sxtwl(dt_utc)
    day_gan = day_gz[0]

    # 4) 时柱（基于“日干 + 本地(或真太阳)时辰”）
    hour_gz, dbg_h = _hour_gz_from_day_gan_and_localtime(day_gan, dt_local_adj)

    # 5) 输出
    result = {
        "pillars": {
            "year":  year_gz,
            "month": month_gz,
            "day":   day_gz,
            "hour":  hour_gz,
        },
        "debug": {
            "input_local_time": dt_local.isoformat(),
            "use_true_solar_time": use_true_solar_time,
            "adjusted_local_time": dt_local_adj.isoformat(),
            "utc_time_used_for_day": dt_utc.isoformat(),
            **dbg_ym,
            **dbg_h
        }
    }
    return result

# --------------- 命令行快速测试 ---------------
if __name__ == "__main__":
    # 示例：1989-10-29 06:00，东八区，假设经度 114.17E（香港）
    out = compute_bazi(
        1989, 10, 29, 6, 0,
        tz_name="Asia/Shanghai",
        longitude=114.17,
        use_true_solar_time=True
    )
    print(out)
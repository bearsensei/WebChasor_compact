# --- Add below your current imports in synthesizer.py ---

import re
import unicodedata

# 常见简体->繁体/繁体->简体特征对（只需几十个高频即可）
_HANS_SIGNS = set([
    "发","国","里","台","后","为","网","体","门","风","长","开","时","东","马","电","苏","万","艺","麦","复","庄","饮","习","叶","罗","齐","画","广","达","杂","报","医","厂","买","卖","书","会","车","云","礼","乐","爱","当","气","汉","龙"
])
_HANT_SIGNS = set([
    "發","國","裡","臺","後","為","網","體","門","風","長","開","時","東","馬","電","蘇","萬","藝","麥","復","莊","飲","習","葉","羅","齊","畫","廣","達","雜","報","醫","廠","買","賣","書","會","車","雲","禮","樂","愛","當","氣","漢","龍"
])

# 高频粤语（书写口语）功能词/语气词/固定搭配
_YUE_TOKENS = [
    "唔","咗","嘅","喺","冇","嚟","咁","嗰","啲","佢","佢哋","喎","啱","嗌","畀","乜","乜嘢","边度","邊度","边个","邊個",
    "点样","點樣","係咪","喺邊","做咩","点解","點解","得嘞","得啦","咪啦","冇啦","算啦","掂","劲","勁","嚟緊","嚟㗎",
    "呀","啦","喇","囉","啩","嘛","咧","噃","喔"  # 句末语气词
]

_YUE_PATTERN = re.compile("|".join(map(re.escape, sorted(_YUE_TOKENS, key=len, reverse=True))))

def _ratio_hant_hans(text: str):
    hans = sum(1 for ch in text if ch in _HANS_SIGNS)
    hant = sum(1 for ch in text if ch in _HANT_SIGNS)
    total = hans + hant
    return (hant / total if total else 0.0, hans / total if total else 0.0)

def _is_ascii_english(text: str):
    letters = sum(1 for ch in text if 'A' <= ch <= 'Z' or 'a' <= ch <= 'z')
    cjk = sum(1 for ch in text if unicodedata.name(ch, "").startswith(("CJK UNIFIED", "CJK COMPATIBILITY")))
    return letters > 0 and cjk == 0

def detect_lang_4way(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "zh-Hans"  # 默认回退

    # 英文优先判定（纯英文或英文占主导且无CJK）
    if _is_ascii_english(q):
        return "en"

    # 简繁粗判
    hant_ratio, hans_ratio = _ratio_hant_hans(q)
    # 小阈值避免少量命中干扰
    if hant_ratio - hans_ratio > 0.05:
        # 进一步粤语检测
        yue_hits = len(_YUE_PATTERN.findall(q))
        # 使用命中次数与长度比做阈值；也可加入「的/了/著/還/因為」等普通话书面词的反特征
        if yue_hits >= 1:
            return "yue-Hant-HK"
        else:
            return "zh-Hant"
    elif hans_ratio - hant_ratio > 0.05:
        return "zh-Hans"
    else:
        # 若难分：根据常见功能词做微调
        # 普通话（繁体书面）常见：沒有、因為、所以、以及、或者、這個、那麼、如果
        mandarin_hant_signals = ["沒有","因為","所以","以及","或者","這個","那麼","如果","比較","的是","了","著","還","與"]
        if any(tok in q for tok in mandarin_hant_signals):
            return "zh-Hant"

        # 粤语信号再试一次
        if _YUE_PATTERN.search(q):
            return "yue-Hant-HK"

        # 回退：用汉字覆盖率估计（包含大量简体字就判简体，否则繁体）
        # 统计繁体字形码位命中更多则选繁体
        return "zh-Hant" if hant_ratio >= hans_ratio else "zh-Hans"

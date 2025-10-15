# src/cache_memory.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from urllib.parse import urlparse
import json, os, time, threading, atexit, shutil

# ---------- 数据模型 ----------
@dataclass(frozen=True)
class Key:
    entity: str   # 规范实体名；无实体用 "_none_"
    task: str     # 任务类别：人物时间线/职位任期/公司财报/地点时区…
    locale: str   # "zh-CN"/"en-US"

@dataclass
class BranchStats:
    trials: int = 0
    wins: int = 0
    avg_score: float = 0.0

@dataclass
class Memo:
    # 线索：去哪找 + 怎么问 + 哪条路先走
    preferred_domains: List[str] = field(default_factory=list)
    blocked_domains:   List[str] = field(default_factory=list)
    good_queries:      List[str] = field(default_factory=list)     # ≤5词的短查询
    event_terms:       List[str] = field(default_factory=list)     # 任命/就任/卸任/appointed/…
    ask_templates:     List[str] = field(default_factory=list)     # 追问句式（Reflexion 用）
    branch_stats: Dict[str, BranchStats] = field(default_factory=dict)  # official/wiki/media
    official_hits:     List[str] = field(default_factory=list)
    score: float = 0.6
    expire_at: float = 0.0

# ---------- 主类：内存 + 文件 ----------
class FileBackedMemory:
    """
    进程内短期记忆 + 文件快照；单进程友好。
    只暴露 3 个方法：get_hints / note_success / note_failure
    """
    def __init__(self, path: str = "./memory_snapshot.json",
                 ttl_seconds: int = 6*3600,
                 caps: Dict[str,int] = None,
                 save_every_ops: int = 10):
        self.path = path
        self.path_bak = f"{path}.bak"
        self.ttl = ttl_seconds
        self.caps = caps or {"preferred_domains":10, "good_queries":20, "official_hits":5}
        self.save_every_ops = max(1, save_every_ops)
        self._lock = threading.RLock()
        self._store: Dict[str, Dict] = {}   # key_str -> memo(as dict)
        self._ops = 0
        self._load()                        # 尝试从文件恢复
        atexit.register(self.save)          # 退出刷盘

    # ---------- 对外：补救前取线索 ----------
    def get_hints(self, entity: str, task: str, locale: str = "zh-CN") -> Dict:
        k_entity = self._ks(entity or "_none_", task, locale)
        k_global = self._ks("_none_", task, locale)
        with self._lock:
            self._gc_locked()
            m_entity = self._to_memo(self._store.get(k_entity))
            m_global = self._to_memo(self._store.get(k_global))
            memo = self._merge_memo(m_entity, m_global)
            # 产出“干净”的提示
            return {
              "preferred_domains": memo.preferred_domains[: self.caps["preferred_domains"]],
              "blocked_domains":   memo.blocked_domains[: 10],
              "good_queries":      memo.good_queries[: self.caps["good_queries"]],
              "event_terms":       memo.event_terms[: 12],
              "ask_templates":     memo.ask_templates[: 10],
              "branch_order":      self._rank_branches(memo.branch_stats)
            }

    # ---------- 对外：补救后记成功 ----------
    def note_success(self, entity: str, task: str, locale: str,
                     used_queries: List[str],
                     official_url: Optional[str] = None,
                     event_terms: Optional[List[str]] = None,
                     branch_name: Optional[str] = None,
                     branch_score: Optional[float] = None):
        key = self._ks(entity or "_none_", task, locale)
        with self._lock:
            self._gc_locked()
            memo = self._to_memo(self._store.get(key)) or Memo()
            memo.expire_at = time.time() + self.ttl
            # 官方命中
            if official_url:
                dom = self._domain_of(official_url)
                self._add(memo.preferred_domains, dom, cap=self.caps["preferred_domains"])
                self._add(memo.official_hits, official_url, cap=self.caps["official_hits"])
            # 短查询/事件词
            for q in used_queries or []:
                if self._word_count(q) <= 5:
                    self._add(memo.good_queries, self._normalize(q), cap=self.caps["good_queries"])
            for e in event_terms or []:
                self._add(memo.event_terms, self._normalize(e))
            # 分支胜率
            if branch_name:
                bs = memo.branch_stats.get(branch_name, BranchStats())
                bs.trials += 1; bs.wins += 1
                if branch_score is not None:
                    bs.avg_score = ((bs.avg_score*(bs.trials-1)) + branch_score) / bs.trials
                memo.branch_stats[branch_name] = bs
            memo.score = min(0.95, memo.score + 0.05)
            self._clip(memo)
            self._store[key] = self._from_memo(memo)
            self._bump_ops_maybe_save()

    # ---------- 对外：补救后记失败 ----------
    def note_failure(self, entity: str, task: str, locale: str,
                     reason: str, bad_domains: Optional[List[str]] = None):
        key = self._ks(entity or "_none_", task, locale)
        with self._lock:
            self._gc_locked()
            memo = self._to_memo(self._store.get(key)) or Memo()
            memo.expire_at = time.time() + self.ttl
            self._add(memo.ask_templates, self._normalize(f"避免：{reason}"))  # 也可写到 bad_patterns
            for d in bad_domains or []:
                self._add(memo.blocked_domains, d, cap=20)
            memo.score = max(0.2, memo.score - 0.03)
            self._clip(memo)
            self._store[key] = self._from_memo(memo)
            self._bump_ops_maybe_save()

    # ---------- 保存/加载 ----------
    def save(self):
        with self._lock:
            self._gc_locked()
            tmp = f"{self.path}.tmp"
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            # 备份旧文件
            if os.path.exists(self.path):
                shutil.copyfile(self.path, self.path_bak)
            # 原子写入
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version":1, "saved_at": time.time(), "items": self._store}, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            self._ops = 0

    # ---------- 内部：加载/清理 ----------
    def _load(self):
        def _try_load(p):
            if not os.path.exists(p): return False
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "items" in data and isinstance(data["items"], dict):
                self._store = data["items"]
                return True
            return False
        ok = _try_load(self.path)
        if not ok:
            _try_load(self.path_bak)
        # 启动即清理一次
        with self._lock:
            self._gc_locked()

    def _gc_locked(self):
        now = time.time()
        to_del = []
        for k, v in self._store.items():
            exp = v.get("expire_at", 0)
            if exp and exp < now:
                to_del.append(k)
        for k in to_del:
            self._store.pop(k, None)

    # ---------- 内部：合并/排序/裁剪 ----------
    def _merge_memo(self, a: Optional[Memo], b: Optional[Memo]) -> Memo:
        out = Memo(expire_at=time.time()+self.ttl)
        for m in (a, b):
            if not m: continue
            for x in m.preferred_domains: self._add(out.preferred_domains, x, cap=self.caps["preferred_domains"])
            for x in m.blocked_domains:   self._add(out.blocked_domains, x, cap=20)
            for x in m.good_queries:      self._add(out.good_queries, x, cap=self.caps["good_queries"])
            for x in m.event_terms:       self._add(out.event_terms, x, cap=50)
            for x in m.ask_templates:     self._add(out.ask_templates, x, cap=50)
            for name, bs in (m.branch_stats or {}).items():
                agg = out.branch_stats.get(name, BranchStats())
                # 累加 trials/wins，avg_score 简单合并
                total = agg.trials + bs.trials
                if total > 0:
                    agg.avg_score = ((agg.avg_score*agg.trials) + (bs.avg_score*bs.trials)) / total
                agg.trials += bs.trials; agg.wins += bs.wins
                out.branch_stats[name] = agg
        self._clip(out)
        return out

    def _rank_branches(self, stats: Dict[str, BranchStats]) -> List[str]:
        if not stats: return ["official","wiki","media"]
        scored = []
        for name, bs in stats.items():
            winrate = (bs.wins / bs.trials) if bs.trials else 0.0
            scored.append((name, 0.7*bs.avg_score + 0.3*winrate))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [n for n,_ in scored]

    def _clip(self, m: Memo):
        m.preferred_domains = m.preferred_domains[: self.caps["preferred_domains"]]
        m.good_queries      = m.good_queries[: self.caps["good_queries"]]
        m.official_hits     = m.official_hits[: self.caps["official_hits"]]

    # ---------- 工具 ----------
    def _ks(self, entity: str, task: str, locale: str) -> str:
        return f"{entity.lower()}|{task}|{locale}"

    def _to_memo(self, obj: Optional[Dict]) -> Optional[Memo]:
        if not obj: return None
        # branch_stats 子结构还原
        bs_raw = obj.get("branch_stats", {})
        bs = {k: BranchStats(**v) for k, v in bs_raw.items()} if isinstance(bs_raw, dict) else {}
        return Memo(
            preferred_domains=list(obj.get("preferred_domains", [])),
            blocked_domains=list(obj.get("blocked_domains", [])),
            good_queries=list(obj.get("good_queries", [])),
            event_terms=list(obj.get("event_terms", [])),
            ask_templates=list(obj.get("ask_templates", [])),
            branch_stats=bs,
            official_hits=list(obj.get("official_hits", [])),
            score=float(obj.get("score", 0.6)),
            expire_at=float(obj.get("expire_at", 0.0))
        )

    def _from_memo(self, m: Memo) -> Dict:
        return {
            "preferred_domains": m.preferred_domains,
            "blocked_domains": m.blocked_domains,
            "good_queries": m.good_queries,
            "event_terms": m.event_terms,
            "ask_templates": m.ask_templates,
            "branch_stats": {k: asdict(v) for k,v in (m.branch_stats or {}).items()},
            "official_hits": m.official_hits,
            "score": m.score,
            "expire_at": m.expire_at
        }

    def _add(self, lst: List[str], item: Optional[str], cap: Optional[int] = None):
        if not item: return
        item = self._normalize(item)
        if item in lst: return
        lst.append(item)
        if cap and len(lst) > cap:
            # 只保留最新 cap 个
            del lst[0: len(lst) - cap]

    def _normalize(self, s: str) -> str:
        return " ".join((s or "").strip().split())

    def _word_count(self, q: str) -> int:
        return len(q.split()) if all(ord(c) < 128 for c in q) else len(q)

    def _domain_of(self, url: str) -> Optional[str]:
        try:
            host = (urlparse(url).hostname or "").lower()
            parts = host.split(".")
            return ".".join(parts[-3:]) if len(parts) >= 3 else host
        except Exception:
            return None

    def _bump_ops_maybe_save(self):
        self._ops += 1
        if self._ops >= self.save_every_ops:
            self.save()
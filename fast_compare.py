# -*- coding: utf-8 -*-
# Fast MOCVD Recipe Comparator (Plotly, event-based)
# + 산점도 피처: Peak ReactorTemp, Pre-Stabilization(없으면 Pre-loop) ReactorPress
# + 산점도 라벨: 파일명 맨 앞 숫자(run number)만 표시

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterable
from functools import lru_cache

import plotly.graph_objects as go
import pandas as pd

DURATION_RE = re.compile(r'^(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})\s*')
COMMENT_RE  = re.compile(r'^\s*"(?P<comment>[^"]*)"\s*,?')
ACTION_RE   = re.compile(r'\s*(?P<var>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*(?P<op>=|to)\s*(?P<val>[^,;]+)\s*(?:,|;)?')

# 키워드 탐지
LOOP_KW_RE  = re.compile(r'\bloop\s+\d+\s*\{', re.IGNORECASE)
STAB_TXT_RE = re.compile(r'stabil', re.IGNORECASE)  # Stabilization/ize/isation 등 포괄

# 파일명 맨 앞 숫자(run number) 추출
RUN_NO_HEAD = re.compile(r'^\s*(\d+)')
def extract_run_no(name: str) -> str:
    m = RUN_NO_HEAD.match(name)
    if m: return m.group(1)
    m2 = re.search(r'(\d+)', name)
    return m2.group(1) if m2 else name

C_TRUE  = {'on','open','enable','enabled','start','true','high'}
C_FALSE = {'off','close','closed','disable','disabled','stop','false','low'}

def parse_hms(tok: str) -> int:
    m = DURATION_RE.match(tok)
    if not m: raise ValueError(f"Invalid time token: {tok}")
    return int(m.group('h'))*3600 + int(m.group('m'))*60 + int(m.group('s'))

def strip_semicolon(s: str) -> str:
    return s[:-1] if s.strip().endswith(';') else s

def boolish(v: Any) -> Optional[int]:
    if isinstance(v, bool): return 1 if v else 0
    if isinstance(v, (int, float)): return None
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in C_TRUE: return 1
        if lv in C_FALSE: return 0
    return None

# --------------------------
# Loop expander
# --------------------------
def expand_loops(text: str) -> str:
    loop_pat = re.compile(r'\bloop\s+(\d+)\s*\{', re.IGNORECASE)
    def _expand(s: str) -> str:
        out, i, L = [], 0, len(s)
        while i < L:
            m = loop_pat.search(s, i)
            if not m:
                out.append(s[i:]); break
            out.append(s[i:m.start()])
            count = int(m.group(1))
            j = m.end(); depth = 1
            while j < L and depth > 0:
                if s[j] == '{': depth += 1
                elif s[j] == '}': depth -= 1
                j += 1
            inner = s[m.end(): j-1]
            out.append(_expand(inner) * count)
            i = j
        return ''.join(out)
    return _expand(text)

# --------------------------
# Parser (lenient)
# --------------------------
@dataclass
class Action:
    var: str
    op: str
    raw_value: str
    value: Any = None
    def parse_value(self):
        v = self.raw_value.strip(); lv = v.lower()
        if lv in C_TRUE:  self.value = True;  return
        if lv in C_FALSE: self.value = False; return
        m = re.match(r'^[+-]?(\d+(\.\d*)?|\.\d+)', v)
        self.value = float(m.group(0)) if m else v

@dataclass
class Step:
    dur_s: int
    comment: Optional[str]
    actions: List[Action] = field(default_factory=list)

@dataclass
class Recipe:
    steps: List[Step] = field(default_factory=list)

class Parser:
    def __init__(self, tolerate_missing_semicolon: bool = True):
        self.tolerate_missing_semicolon = tolerate_missing_semicolon

    def parse(self, text: str) -> Recipe:
        cleaned = self._preclean(text)
        expanded = expand_loops(cleaned)
        blocks = self._gather(expanded)
        steps: List[Step] = []
        for b in blocks:
            b_strip = b.strip()
            if not b_strip or not DURATION_RE.match(b_strip):
                continue
            steps.append(self._parse_block(b))
        return Recipe(steps)

    def _preclean(self, text: str) -> str:
        out_lines = []
        for raw in text.splitlines():
            line = raw.rstrip("\n")
            s = line.strip()
            if not s: continue
            if s.startswith("#") or s.startswith("//"): continue
            if all(ch in "#-=*" for ch in s): continue
            if "#" in line:
                line = line.split("#", 1)[0].rstrip()
                if not line.strip(): continue
            out_lines.append(line)
        return "\n".join(out_lines)

    def _gather(self, text: str) -> List[str]:
        blocks, buf = [], []
        for raw in text.splitlines():
            line = raw.rstrip()
            if not line.strip(): continue
            buf.append(line)
            if line.strip().endswith(';'):
                blocks.append('\n'.join(buf)); buf = []
        if buf:
            if self.tolerate_missing_semicolon:
                blocks.append('\n'.join(buf))
            else:
                raise ValueError("Last block missing ';'")
        return blocks

    def _parse_block(self, block: str) -> Step:
        s = block.strip()
        m = DURATION_RE.match(s); dur = parse_hms(s); s = s[m.end():].lstrip()
        comment=None
        m2 = COMMENT_RE.match(s)
        if m2:
            comment = m2.group('comment').strip()
            s = s[m2.end():].lstrip()
        s = strip_semicolon(s)

        actions, i = [], 0
        while i < len(s):
            m3 = ACTION_RE.match(s, i)
            if not m3:
                if i < len(s) and s[i] in {',',' '}: i += 1; continue
                break
            a = Action(var=m3.group('var').strip(), op=m3.group('op').strip(),
                       raw_value=m3.group('val').strip())
            a.parse_value(); actions.append(a); i = m3.end()
        return Step(dur_s=dur, comment=comment, actions=actions)

# --------------------------
# Change-point builder (no dt)
# --------------------------
def build_change_points(recipe: Recipe, variables: Iterable[str]) -> Tuple[Dict[str, List[Tuple[float, float]]], float]:
    wanted = set(variables)
    state: Dict[str, Any] = {}
    series_cp: Dict[str, List[Tuple[float, float]]] = {v: [] for v in wanted}

    t_cursor = 0.0
    for st in recipe.steps:
        t0, t1 = t_cursor, t_cursor + st.dur_s

        for a in st.actions:
            if a.var not in wanted: continue
            if a.op == '=':
                val = a.value
                b = boolish(val); val = b if b is not None else val
                prev = state.get(a.var, None)
                if prev is not None and len(series_cp[a.var])==0:
                    series_cp[a.var].append((t0, float(prev) if isinstance(prev,(int,float)) else (1.0 if prev else 0.0)))
                vnum = float(val) if isinstance(val,(int,float)) else (1.0 if val else 0.0)
                series_cp[a.var].append((t0, vnum))
                state[a.var] = val

        for a in st.actions:
            if a.var not in wanted: continue
            if a.op == 'to':
                target = a.value
                b = boolish(target); target = b if b is not None else target
                prev = state.get(a.var, None)
                if isinstance(target,(int,float)) and isinstance(prev,(int,float)):
                    if not series_cp[a.var] or series_cp[a.var][-1][0] < t0:
                        series_cp[a.var].append((t0, float(prev)))
                    series_cp[a.var].append((t1, float(target)))
                    state[a.var] = float(target)
                else:
                    vnum = float(target) if isinstance(target,(int,float)) else (1.0 if target else 0.0)
                    series_cp[a.var].append((t1, vnum))
                    state[a.var] = target
        t_cursor = t1

    total_T = t_cursor
    for var, pts in series_cp.items():
        pts.sort(key=lambda x: x[0])
        cleaned, last_v = [], None
        for (t, v) in pts:
            if last_v is not None and abs(v - last_v) < 1e-12:
                continue
            cleaned.append((t, v)); last_v = v
        series_cp[var] = cleaned
    return series_cp, total_T

# --------------------------
# Cache by content
# --------------------------
@lru_cache(maxsize=256)
def _parse_cached(text: str) -> Recipe:
    parser = Parser(tolerate_missing_semicolon=True)
    return parser.parse(text)

# --------------------------
# Batch compare (기존)
# --------------------------
def compare_memory(files: List[Tuple[str, str]], vars: List[str], align_zero: bool=True) -> Dict[str, "go.Figure"]:
    runs = []
    for name, txt in files:
        recipe = _parse_cached(txt)
        series_cp, total_T = build_change_points(recipe, vars)
        runs.append({"name": name, "series_cp": series_cp, "total_T": total_T})

    figs: Dict[str, go.Figure] = {}
    for var in vars:
        fig = go.Figure()
        for r in runs:
            if var not in r["series_cp"]: continue
            pts = r["series_cp"][var]
            if not pts: continue
            base = pts[0][0] if align_zero else 0.0
            xs = [t - base for (t, _) in pts]
            ys = [v for (_, v) in pts]
            finite = [v for v in ys if v is not None]
            is_binary = bool(finite) and set(finite).issubset({0.0,1.0})
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines', name=r["name"],
                line_shape='hv' if is_binary else 'linear'
            ))
        fig.update_layout(
            title=f"Compare: {var}",
            xaxis_title="Time (s)" + (" (t0-aligned)" if align_zero else ""),
            yaxis_title=var,
            template="plotly_white",
            legend_title="run"
        )
        figs[var] = fig
    return figs

def tidy_memory(files: List[Tuple[str, str]], vars: List[str], align_zero: bool=True) -> pd.DataFrame:
    rows=[]
    for name, txt in files:
        recipe = _parse_cached(txt)
        series_cp, _ = build_change_points(recipe, vars)
        for var, pts in series_cp.items():
            if not pts: continue
            base = pts[0][0] if align_zero else 0.0
            for (t, v) in pts:
                rows.append({"run": name, "variable": var, "time_s": t - base, "value": v})
    return pd.DataFrame(rows)

# --------------------------
# NEW: 산점도용 피처 (Peak T, Pre-Ref P: Stabilization≻Loop)
# --------------------------
def _preclean_text(text: str) -> str:
    out_lines = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        s = line.strip()
        if not s: continue
        if s.startswith("#") or s.startswith("//"): continue
        if all(ch in "#-=*" for ch in s): continue
        if "#" in line:
            line = line.split("#", 1)[0].rstrip()
            if not line.strip(): continue
        out_lines.append(line)
    return "\n".join(out_lines)

def _first_stabilization_time(recipe: Recipe) -> Optional[float]:
    t = 0.0
    for st in recipe.steps:
        comment = (st.comment or "").lower()
        if STAB_TXT_RE.search(comment):
            return t  # 해당 스텝 시작 시각
        t += st.dur_s
    return None

def _peak_temp_from_text(text: str) -> Optional[float]:
    recipe = _parse_cached(text)
    series_cp, _ = build_change_points(recipe, ["ReactorTemp"])
    pts = series_cp.get("ReactorTemp", [])
    if not pts: return None
    vals = [v for (_, v) in pts if v is not None]
    return max(vals) if vals else None

def _press_before_time(recipe: Recipe, t_cut: float) -> Optional[float]:
    series_cp, _ = build_change_points(recipe, ["ReactorPress"])
    pts = series_cp.get("ReactorPress", [])
    if not pts: return None
    prev_val = None
    for (t, v) in pts:
        if t < t_cut:
            prev_val = v
        else:
            break
    return prev_val

def _preloop_press_from_text(text: str) -> Optional[float]:
    cleaned = _preclean_text(text)
    m = LOOP_KW_RE.search(cleaned)
    pre = cleaned[:m.start()] if m else cleaned
    recipe = _parse_cached(pre)
    series_cp, _ = build_change_points(recipe, ["ReactorPress"])
    pts = series_cp.get("ReactorPress", [])
    if not pts: return None
    return pts[-1][1]

def _pre_stabilization_or_loop_press(text: str) -> Optional[float]:
    recipe = _parse_cached(text)
    t_stab = _first_stabilization_time(recipe)
    if t_stab is not None:
        val = _press_before_time(recipe, t_stab)
        if val is not None:
            return val
    return _preloop_press_from_text(text)

def scatter_features_memory(files: List[Tuple[str, str]]) -> Tuple[pd.DataFrame, "go.Figure"]:
    rows = []
    for name, txt in files:
        x = _peak_temp_from_text(txt)
        y = _pre_stabilization_or_loop_press(txt)
        rows.append({
            "run": name,
            "run_no": extract_run_no(name),    # 라벨용 run number
            "ReactorTemp_peak": x,
            "ReactorPress_preRef": y
        })
    df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ReactorTemp_peak"],
        y=df["ReactorPress_preRef"],
        mode="markers+text",
        text=df["run_no"],                  # 숫자 라벨만
        textposition="top center",
        name="runs",
        hovertemplate="run=%{text}<br>PeakT=%{x}<br>PreRefP=%{y}<extra></extra>"
    ))
    fig.update_layout(
        title="Peak ReactorTemp  vs  Pre-Ref ReactorPress (label: run#)",
        xaxis_title="Peak ReactorTemp",
        yaxis_title="Pre-Ref ReactorPress",
        template="plotly_white"
    )
    return df, fig

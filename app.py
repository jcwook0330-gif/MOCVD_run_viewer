# -*- coding: utf-8 -*-
# MOCVD Recipe Visualizer (Streamlit)
# - 단일 레시피(겹쳐/분리 플롯)
# - 배치 비교(Plotly)
# - 산점도(라벨=run#)
# - 루프 분석(요약/상세)
# - NEW: 모든 시간축 플롯에 루프 구간 밴드 표시

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from fast_compare import (
    compare_memory,
    tidy_memory,
    scatter_features_memory,
    loops_summary_memory,
    loop_windows_seconds,   # ← 루프 밴드 계산
)

# -------- 공통 정규식/헬퍼/파서/타임라인 (생략 없이 기존 코드 유지) --------
DURATION_RE = re.compile(r'^(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})\s*')
COMMENT_RE  = re.compile(r'^\s*"(?P<comment>[^"]*)"\s*,?')
ACTION_RE   = re.compile(r'\s*(?P<var>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*(?P<op>=|to)\s*(?P<val>[^,;]+)\s*(?:,|;)?')

C_TRUE  = {'on','open','enable','enabled','start','true','high'}
C_FALSE = {'off','close','closed','disable','disabled','stop','false','low'}

def parse_hms(tok: str) -> int:
    m = DURATION_RE.match(tok)
    if not m: raise ValueError(f"Invalid time token: {tok}")
    return int(m.group('h'))*3600 + int(m.group('m'))*60 + int(m.group('s'))

def strip_semicolon(s: str) -> str:
    return s[:-1] if s.strip().endswith(';') else s

def to_boolish(v: Any) -> Optional[int]:
    if isinstance(v, bool): return 1 if v else 0
    if isinstance(v, (int, float)): return None
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in C_TRUE: return 1
        if lv in C_FALSE: return 0
    return None

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
    time_s: int
    comment: Optional[str]
    actions: List[Action] = field(default_factory=list)

@dataclass
class Recipe:
    steps: List[Step] = field(default_factory=list)

def expand_loops_with_blocks(text: str):
    loop_pat = re.compile(r'\bloop\s+(\d+)\s*\{', re.IGNORECASE)
    def _expand(s: str, next_id: int = 1):
        out_parts, metas = [], []
        i, L = 0, len(s)
        while i < L:
            m = loop_pat.search(s, i)
            if not m:
                out_parts.append(s[i:]); break
            out_parts.append(s[i:m.start()])
            count = int(m.group(1))
            j = m.end(); depth = 1
            while j < L and depth > 0:
                if s[j] == '{': depth += 1
                elif s[j] == '}': depth -= 1
                j += 1
            inner = s[m.end(): j-1]
            inner_expanded, inner_meta, next_id = _expand(inner, next_id)
            metas.append({"id": next_id, "count": count, "block_text": inner_expanded})
            next_id += 1
            out_parts.append(inner_expanded * count)
            metas.extend(inner_meta)
            i = j
        return "".join(out_parts), metas, next_id
    expanded, meta, _ = _expand(text, 1)
    return expanded, meta

class Parser:
    def __init__(self, tolerate_missing_semicolon: bool = True):
        self.tolerate_missing_semicolon = tolerate_missing_semicolon
        self.loop_blocks: List[Dict[str, Any]] = []
    def parse(self, text: str) -> Recipe:
        cleaned = self._preclean(text)
        expanded, loop_blocks = expand_loops_with_blocks(cleaned)
        self.loop_blocks = loop_blocks or []
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
        m = DURATION_RE.match(s);  t = parse_hms(s);  s = s[m.end():].lstrip()
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
        return Step(time_s=t, comment=comment, actions=actions)

class Timeline:
    def __init__(self, dt:int=1, absolute:bool=False):
        if dt <= 0: raise ValueError("dt must be >= 1")
        self.dt = dt; self.absolute = absolute
    def build(self, recipe: Recipe) -> Tuple[List[int], Dict[str, List[Any]], List[Tuple[int, int, Step]]]:
        windows: List[Tuple[int, int, Step]] = []
        cursor = 0
        for st in recipe.steps:
            if self.absolute:
                t0 = cursor; t1 = st.time_s
                if t1 < t0: raise ValueError("Absolute times must be non-decreasing.")
                cursor = t1
            else:
                t0 = cursor; t1 = cursor + st.time_s; cursor = t1
            windows.append((t0, t1, st))
        total_T = windows[-1][1] if windows else 0
        times = list(range(0, total_T + 1, self.dt))
        series: Dict[str, List[Any]] = {}; state: Dict[str, Any] = {}
        for (t0, t1, st) in windows:
            ramps: List[Tuple[str, float, float]] = []; jumps: List[Tuple[str, Any]] = []
            for a in st.actions:
                if a.op == '=':
                    val = a.value; b = to_boolish(val); val = b if b is not None else val
                    jumps.append((a.var, val))
                elif a.op == 'to':
                    prev = state.get(a.var); val = a.value; b = to_boolish(val); val = b if b is not None else val
                    if isinstance(prev,(int,float)) and isinstance(val,(int,float)):
                        ramps.append((a.var, float(prev), float(val)))
                    else:
                        jumps.append((a.var, val))
            for var, val in jumps: state[var] = val
            need_vars = set(state.keys()) | {v for (v,_,_) in ramps}
            for var in need_vars:
                if var not in series: series[var] = [None]*len(times)
            for idx, t in enumerate(times):
                if t < t0 or t > t1: continue
                for var, v0, v1 in ramps:
                    vt = v1 if t1 == t0 else (v0 + ((t-t0)/(t1-t0))*(v1-v0))
                    state[var] = vt
                for var in series.keys():
                    val = state.get(var, series[var][idx-1] if idx>0 else None)
                    series[var][idx] = val
        for var, arr in series.items():
            last = None
            for i, v in enumerate(arr):
                if v is None and last is not None: arr[i] = last
                elif v is not None: last = v
        return times, series, windows

# --------------------------
# Matplotlib: 루프 밴드 오버레이
# --------------------------
def add_loop_spans(ax, windows: List[Dict[str, float]]):
    if not windows: return
    for w in windows:
        ax.axvspan(w["start"], w["end"], alpha=0.12, color='orange')
        # x: 데이터좌표, y: 축비율(상단 98%)
        xmid = 0.5*(w["start"]+w["end"])
        ax.text(xmid, 0.98, f"loop×{int(w['count'])}", transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8, color='dimgray')

def list_variables(series: Dict[str, List[Any]]) -> List[str]:
    return sorted(series.keys())

def to_numeric_array(arr: List[Any]) -> np.ndarray:
    out = []
    for v in arr:
        b = to_boolish(v)
        if b is not None: out.append(float(b))
        elif isinstance(v,(int,float)): out.append(float(v))
        else: out.append(np.nan)
    return np.array(out, dtype=float)

def plot_overlay(times, series, vars_to_plot, loop_windows=None):
    plt.figure()
    ax = plt.gca()
    for var in vars_to_plot:
        if var not in series: continue
        y = to_numeric_array(series[var])
        finite = y[~np.isnan(y)]
        if finite.size and set(np.unique(finite)).issubset({0.0,1.0}):
            ax.step(times, y, where="post", label=var)
        else:
            ax.plot(times, y, label=var)
    if loop_windows: add_loop_spans(ax, loop_windows)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(", ".join(vars_to_plot))
    ax.set_title(" / ".join(vars_to_plot)); ax.grid(True); ax.legend()
    st.pyplot(plt.gcf()); plt.close()

def plot_separate(times, series, vars_to_plot, loop_windows=None):
    for var in vars_to_plot:
        if var not in series: continue
        plt.figure(); ax = plt.gca()
        y = to_numeric_array(series[var])
        finite = y[~np.isnan(y)]
        if finite.size and set(np.unique(finite)).issubset({0.0,1.0}):
            ax.step(times, y, where="post", label=var)
        else:
            ax.plot(times, y, label=var)
        if loop_windows: add_loop_spans(ax, loop_windows)
        ax.set_xlabel("Time (s)"); ax.set_ylabel(var); ax.set_title(var)
        ax.grid(True); ax.legend(); st.pyplot(plt.gcf()); plt.close()

# --------------------------
# UI (탭 레이아웃) — 필요한 부분만 발췌
# --------------------------
st.set_page_config(page_title="MOCVD Recipe Viewer", layout="wide")
st.title("📈 MOCVD 레시피 뷰어")

with st.sidebar:
    st.subheader("📂 공용 업로드")
    _files_shared = st.file_uploader("여러 레시피(.txt)", type=["txt"], accept_multiple_files=True, key="multi_shared")
    if _files_shared:
        st.session_state["batch_files"] = [(f.name, f.read().decode("utf-8", errors="ignore")) for f in _files_shared]
    st.caption("업로드하면 ‘배치 비교/산점도/루프 분석’ 탭에서 그대로 사용됩니다.")

tab_single, tab_batch, tab_scatter, tab_loop = st.tabs(["단일", "배치 비교", "산점도", "루프 분석"])

# ============== 탭 1: 단일 ==============
with tab_single:
    st.subheader("단일 레시피")
    dt = st.number_input("샘플링 간격 dt (s)", min_value=1, value=1, step=1, key="dt_single")
    absolute = st.checkbox("타임스탬프 절대 해석", value=False, key="abs_single")
    mode = st.radio("플롯 모드", ["겹쳐 그리기(한 그림)", "변수별 분리"], horizontal=True, key="mode_single")
    show_loop_band = st.checkbox("루프 구간 표시", value=True, key="loop_band_single")

    up = st.file_uploader("레시피(.txt)", type=["txt"], key="single_up")
    use_demo = st.checkbox("내장 데모", value=False, key="demo_single")

    if up or use_demo:
        if use_demo:
            text = (
                '0:00:05 "Stabilization";\n'
                'loop 3 {\n'
                '  0:00:02 "TEBo on / NH3 off",  TMGa_2.run = open, DummyMO1.run = close;\n'
                '  0:00:01 "Interruption",        TMGa_2.run = close, DummyMO1.run = open;\n'
                '  0:00:01 "TEBo off / NH3 on",   NH3_1.run = open, RunHydride = 800, PushHydride = 1000;\n'
                '  0:00:01 "Interruption",        NH3_1.run = close, RunHydride = 5000, PushHydride = 5000;\n'
                '}\n'
                '0:00:01 "End growth", TMGa_2.run = close, NH3_1.run = open;\n'
            )
        else:
            text = up.read().decode("utf-8", errors="ignore")

        parser = Parser(True)
        recipe = parser.parse(text)
        times, series, windows = Timeline(dt=dt, absolute=absolute).build(recipe)

        # 루프 밴드 계산
        loop_windows = loop_windows_seconds(text) if show_loop_band else []

        if series:
            vars_all = list_variables(series)
            defaults = [v for v in ["CeilingTemp","ReactorTemp","ReactorPress","RF_U","NH3_1.source","NH3_1.run"] if v in vars_all][:3]
            picked = st.multiselect("시각화할 변수", vars_all, default=defaults, key="single_vars")
            if picked:
                if mode.startswith("겹쳐"):
                    plot_overlay(times, series, picked, loop_windows=loop_windows)
                else:
                    plot_separate(times, series, picked, loop_windows=loop_windows)
        else:
            st.info("파싱 가능한 데이터가 없습니다.")

# ============== 탭 2: 배치 비교 ==============
with tab_batch:
    st.subheader("여러 레시피 비교 (Plotly)")
    file_tuples = st.session_state.get("batch_files", None)
    if not file_tuples:
        st.info("사이드바에서 여러 레시피를 업로드하세요.")
    else:
        # 변수 후보(첫 파일 기준)
        try:
            p0 = Parser(True); r0 = p0.parse(file_tuples[0][1])
            _, s0, _ = Timeline(dt=1, absolute=False).build(r0)
            all_vars = sorted(s0.keys())
        except Exception:
            all_vars = []
        defaults = [v for v in ["CeilingTemp","ReactorTemp","ReactorPress","RF_U","NH3_1.source"] if v in all_vars] or all_vars[:3]
        vars_to_compare = st.multiselect("비교 변수", all_vars, default=defaults, key="cmp_vars_tab")
        align_zero = st.checkbox("t=0 정렬", value=True, key="align0_tab")
        show_loop_band_first = st.checkbox("루프 구간 표시(첫 번째 run 기준)", value=True, key="loop_band_batch")

        if vars_to_compare:
            figs = compare_memory(
                file_tuples, vars=vars_to_compare,
                align_zero=align_zero, show_loop_band_first=show_loop_band_first
            )
            for var, fig in figs.items():
                st.plotly_chart(fig, use_container_width=True)

# ============== 탭 3: 산점도 ==============
with tab_scatter:
    st.subheader("Peak ReactorTemp vs Pre-Ref ReactorPress (라벨=run#)")
    file_tuples = st.session_state.get("batch_files", None)
    if not file_tuples:
        st.info("사이드바에서 여러 레시피를 업로드하세요.")
    else:
        df_feat, fig_scatter = scatter_features_memory(file_tuples)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.dataframe(df_feat, use_container_width=True)

# ============== 탭 4: 루프 분석 ==============
with tab_loop:
    st.subheader("Loop 분석 (요약/상세)")
    file_tuples = st.session_state.get("batch_files", None)
    if not file_tuples:
        st.info("사이드바에서 여러 레시피를 업로드하세요.")
    else:
        df_loops, df_steps = loops_summary_memory(file_tuples)
        st.markdown("**요약표 (파일별 loop)**")
        st.dataframe(df_loops, use_container_width=True)
        st.download_button("CSV 다운로드(Loop 요약)", df_loops.to_csv(index=False).encode("utf-8-sig"),
                           file_name="loops_summary.csv", mime="text/csv")

        st.markdown("**상세표 (1 cycle step-by-step)**")
        if not df_steps.empty:
            runs = sorted(df_steps["run"].unique().tolist())
            pick_run = st.selectbox("Run 선택", runs, index=0, key="loop_run_tab")
            loops_in_run = sorted(df_steps[df_steps["run"]==pick_run]["loop_id"].unique().tolist())
            pick_loop = st.selectbox("Loop 선택", loops_in_run, index=0, key="loop_id_tab")

            view = df_steps[(df_steps["run"]==pick_run) & (df_steps["loop_id"]==pick_loop)].sort_values("step_idx")
            st.dataframe(view, use_container_width=True)

            lines = [f"({int(r.duration_s)}s) comment='{r.comment}' | actions='{r.actions}'" for r in view.itertuples()]
            st.code("\n".join(lines), language="text")

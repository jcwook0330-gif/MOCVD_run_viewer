# -*- coding: utf-8 -*-
# MOCVD Recipe Visualizer (Streamlit)
# - 단일 레시피: 루프 전개 + 루프 요약 표 + 루프 패턴 뷰 + 상세 로그 + 플롯
# - 배치 비교: 여러 파일 업로드 → 변수별 run 비교(Plotly, 이벤트 기반 빠른 렌더)
# - 산점도: Peak ReactorTemp (x) vs Pre-Stabilization(없으면 Pre-loop) ReactorPress (y), 라벨=run#
# - Loop 분석: 파일별 loop 개수/시간 요약 + 1cycle 상세(step-by-step)
# - 주석(#/ // / 구분선) 무시, 마지막 세미콜론 누락 허용, '='(즉시), 'to'(선형 램프)

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 배치 비교 + 산점도 + 루프 분석 유틸
from fast_compare import (
    compare_memory,
    tidy_memory,
    scatter_features_memory,
    loops_summary_memory,
)

# --------------------------
# 정규식 & 헬퍼
# --------------------------
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

# --------------------------
# loop 전개 + 메타(요약용)
# --------------------------
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

# --------------------------
# 데이터 클래스
# --------------------------
@dataclass
class Action:
    var: str
    op: str      # '=' or 'to'
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

# --------------------------
# 파서
# --------------------------
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

# --------------------------
# 타임라인 생성 (단일 레시피용)
# --------------------------
class Timeline:
    def __init__(self, dt:int=1, absolute:bool=False):
        if dt <= 0:
            raise ValueError("dt must be >= 1")
        self.dt = dt
        self.absolute = absolute

    def build(self, recipe: Recipe) -> Tuple[List[int], Dict[str, List[Any]], List[Tuple[int, int, Step]]]:
        windows: List[Tuple[int, int, Step]] = []
        cursor = 0
        for st in recipe.steps:
            if self.absolute:
                t0 = cursor
                t1 = st.time_s
                if t1 < t0:
                    raise ValueError("Absolute times must be non-decreasing.")
                cursor = t1
            else:
                t0 = cursor
                t1 = cursor + st.time_s
                cursor = t1
            windows.append((t0, t1, st))

        total_T = windows[-1][1] if windows else 0
        times = list(range(0, total_T + 1, self.dt))
        series: Dict[str, List[Any]] = {}
        state: Dict[str, Any] = {}

        for (t0, t1, st) in windows:
            ramps: List[Tuple[str, float, float]] = []
            jumps: List[Tuple[str, Any]] = []

            for a in st.actions:
                if a.op == '=':
                    val = a.value
                    b = to_boolish(val)
                    val = b if b is not None else val
                    jumps.append((a.var, val))
                elif a.op == 'to':
                    prev = state.get(a.var)
                    val = a.value
                    b = to_boolish(val)
                    val = b if b is not None else val
                    if isinstance(prev, (int, float)) and isinstance(val, (int, float)):
                        ramps.append((a.var, float(prev), float(val)))
                    else:
                        jumps.append((a.var, val))

            for var, val in jumps:
                state[var] = val

            need_vars = set(state.keys()) | {v for (v, _, _) in ramps}
            for var in need_vars:
                if var not in series:
                    series[var] = [None] * len(times)

            for idx, t in enumerate(times):
                if t < t0 or t > t1:
                    continue
                for var, v0, v1 in ramps:
                    if t1 == t0:
                        vt = v1
                    else:
                        alpha = (t - t0) / (t1 - t0)
                        alpha = 0.0 if alpha < 0 else (1.0 if alpha > 1 else alpha)
                        vt = v0 + alpha * (v1 - v0)
                    state[var] = vt
                for var in series.keys():
                    val = state.get(var, series[var][idx - 1] if idx > 0 else None)
                    series[var][idx] = val

        for var, arr in series.items():
            last = None
            for i, v in enumerate(arr):
                if v is None and last is not None:
                    arr[i] = last
                elif v is not None:
                    last = v

        return times, series, windows

# --------------------------
# 플롯 유틸 (단일 레시피)
# --------------------------
def list_variables(series: Dict[str, List[Any]]) -> List[str]:
    return sorted(series.keys())

def to_numeric_array(arr: List[Any]) -> np.ndarray:
    out = []
    for v in arr:
        b = to_boolish(v)
        if b is not None:
            out.append(float(b))
        elif isinstance(v,(int,float)):
            out.append(float(v))
        else:
            out.append(np.nan)
    return np.array(out, dtype=float)

def plot_overlay(times, series, vars_to_plot):
    plt.figure()
    for var in vars_to_plot:
        if var not in series: continue
        y = to_numeric_array(series[var])
        finite = y[~np.isnan(y)]
        if finite.size and set(np.unique(finite)).issubset({0.0,1.0}):
            plt.step(times, y, where="post", label=var)
        else:
            plt.plot(times, y, label=var)
    plt.xlabel("Time (s)"); plt.ylabel(", ".join(vars_to_plot))
    plt.title(" / ".join(vars_to_plot)); plt.grid(True); plt.legend()
    st.pyplot(plt.gcf()); plt.close()

def plot_separate(times, series, vars_to_plot):
    for var in vars_to_plot:
        if var not in series: continue
        plt.figure()
        y = to_numeric_array(series[var])
        finite = y[~np.isnan(y)]
        if finite.size and set(np.unique(finite)).issubset({0.0,1.0}):
            plt.step(times, y, where="post", label=var)
        else:
            plt.plot(times, y, label=var)
        plt.xlabel("Time (s)"); plt.ylabel(var); plt.title(var)
        plt.grid(True); plt.legend(); st.pyplot(plt.gcf()); plt.close()

# --------------------------
# 루프 패턴 요약 헬퍼 (단일 화면용 텍스트)
# --------------------------
def summarize_loop_steps(block_text: str):
    tmp_parser = Parser(tolerate_missing_semicolon=True)
    tmp_recipe = tmp_parser.parse(block_text)
    items = [(st.time_s, (st.comment or '').strip()) for st in tmp_recipe.steps]
    total_sec = sum(d for d, _ in items)
    return items, total_sec, len(items)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="MOCVD Recipe Visualizer", layout="wide")
st.title("📈 MOCVD 레시피 시각화 (단일 + 배치 비교 + 산점도 + 루프 분석)")

with st.expander("옵션", expanded=True):
    dt = st.number_input("샘플링 간격 dt (s)", min_value=1, value=1, step=1, key="dt")
    absolute = st.checkbox("타임스탬프를 절대 시간으로 해석", value=False, key="abs")
    mode = st.radio("단일 파일 플롯 모드", ["겹쳐 그리기(한 그림)", "변수별 분리"], horizontal=True, key="mode")

# ---- A) 단일 레시피
uploaded = st.file_uploader("단일 레시피 업로드 (.txt)", type=["txt"])
use_demo = st.checkbox("내장 데모 사용", value=False)

if uploaded or use_demo:
    if use_demo:
        text = (
            'loop 3 {\n'
            '  0:00:02 "TEBo on / NH3 off",  TMGa_2.run = open, DummyMO1.run = close;\n'
            '  0:00:01 "Interruption",        TMGa_2.run = close, DummyMO1.run = open;\n'
            '  0:00:01 "TEBo off / NH3 on",   NH3_1.run = open, RunHydride = 800, PushHydride = 1000;\n'
            '  0:00:01 "Interruption",        NH3_1.run = close, RunHydride = 5000, PushHydride = 5000;\n'
            '}\n'
            '0:00:01 "End growth", TMGa_2.run = close, TMGa_2.line = close, DummyMO1.run = open,\n'
            '                      NH3_1.run = open;\n'
        )
    else:
        text = uploaded.read().decode("utf-8", errors="ignore")

    parser = Parser(tolerate_missing_semicolon=True)
    recipe = parser.parse(text)
    times, series, windows = Timeline(dt=dt, absolute=absolute).build(recipe)

    if not series:
        st.warning("파싱 가능한 명령이 없습니다.")
    else:
        vars_all = list_variables(series)
        default_pick = [v for v in ["CeilingTemp","ReactorTemp","ReactorPress","RF_U",
                                    "NH3_1.source","NH3_1.run","TMGa_2.run","DummyMO1.run"]
                        if v in vars_all][:3]
        picked = st.multiselect("시각화할 변수 선택", vars_all, default=default_pick, key="single_vars")

        df_single = pd.DataFrame({"time_s": times, **{k: series[k] for k in vars_all}})
        st.download_button("CSV 다운로드(단일)", data=df_single.to_csv(index=False).encode("utf-8-sig"),
                           file_name="timeline_single.csv", mime="text/csv")

        with st.expander("요약", expanded=True):
            st.write(f"총 스텝(전개 후): {len(recipe.steps)} | 총 시간: {times[-1]} s")
            if parser.loop_blocks:
                st.subheader("Loop 요약")
                rows=[]
                for lb in parser.loop_blocks:
                    tmp_p = Parser(True); tmp_r = tmp_p.parse(lb["block_text"])
                    cycle_steps = len(tmp_r.steps); cycle_dur = sum(s.time_s for s in tmp_r.steps)
                    rows.append({"Loop ID": lb["id"], "Cycles": lb["count"],
                                 "Steps / cycle": cycle_steps, "Sec / cycle": cycle_dur,
                                 "Total sec": cycle_dur * lb["count"]})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with st.expander("루프 패턴 (1 cycle 요약)", expanded=True):
            if parser.loop_blocks:
                for lb in parser.loop_blocks:
                    items, cycle_sec, cycle_steps = summarize_loop_steps(lb["block_text"])
                    st.markdown(f"**loop ({lb['count']})** — one cycle: {cycle_sec}s, {cycle_steps} steps")
                    lines = [f"({dur}s) comment = '{com}'" for dur, com in items]
                    st.code("\n".join(lines), language="text")
                    st.markdown("---")
            else:
                st.info("요약할 loop가 없습니다.")

        with st.expander("전체 스텝 로그 (상세)", expanded=False):
            preview_n = st.slider("미리보기 개수", 10, 200, 50, step=10, key="preview_n_single")
            for i,(t0,t1,stp) in enumerate(windows[:preview_n],1):
                st.text(f"[{i:02d}] {t0:>5}s → {t1:>5}s ({t1-t0:>3}s)  comment='{stp.comment or ''}'")
            if len(windows) > preview_n:
                st.text(f"... (총 {len(windows)}개 중 {preview_n}개 표시)")

        if picked:
            if mode.startswith("겹쳐"):
                plot_overlay(times, series, picked)
            else:
                plot_separate(times, series, picked)

# ---- B) 배치 비교
st.markdown("---")
st.header("🧪 배치 비교 (여러 레시피 업로드)")

files = st.file_uploader("여러 레시피 업로드 (.txt)", type=["txt"], accept_multiple_files=True, key="multi_up")
if files:
    file_tuples = [(f.name, f.read().decode("utf-8", errors="ignore")) for f in files]

    # 변수 후보(간단히 첫 파일 기준)
    all_vars = set()
    try:
        p0 = Parser(True); r0 = p0.parse(file_tuples[0][1])
        _, s0, _ = Timeline(dt=1, absolute=False).build(r0)
        all_vars.update(s0.keys())
    except Exception:
        pass

    all_vars = sorted(all_vars) if all_vars else []
    default_vars = [v for v in ["CeilingTemp","ReactorTemp","ReactorPress","RF_U","NH3_1.source"] if v in all_vars] or all_vars[:3]
    vars_to_compare = st.multiselect("비교할 변수 선택", all_vars, default=default_vars, key="cmp_vars")

    align_zero = st.checkbox("각 run을 t=0으로 정렬(권장)", value=True, key="align0")

    if vars_to_compare:
        figs = compare_memory(file_tuples, vars=vars_to_compare, align_zero=align_zero)
        for var, fig in figs.items():
            st.plotly_chart(fig, use_container_width=True)

        df_tidy = tidy_memory(file_tuples, vars=vars_to_compare, align_zero=align_zero)
        st.download_button("CSV 다운로드(배치 tidy)", data=df_tidy.to_csv(index=False).encode("utf-8-sig"),
                           file_name="batch_tidy.csv", mime="text/csv")

    # 산점도: Peak T vs Pre-Ref P  (라벨=run number)
    st.subheader("📍 분산도: Peak ReactorTemp  vs  Pre-Ref ReactorPress (라벨=run#)")
    df_feat, fig_scatter = scatter_features_memory(file_tuples)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.dataframe(df_feat, use_container_width=True)
    st.download_button("CSV 다운로드(산점도 피처)", data=df_feat.to_csv(index=False).encode("utf-8-sig"),
                       file_name="scatter_features.csv", mime="text/csv")

    # ---------- NEW: Loop 분석 ----------
    with st.expander("🔁 Loop 분석 (요약/상세)", expanded=False):
        df_loops, df_steps = loops_summary_memory(file_tuples)

        st.markdown("**요약표 (파일별 loop)**")
        st.dataframe(df_loops, use_container_width=True)
        st.download_button(
            "CSV 다운로드(Loop 요약)",
            data=df_loops.to_csv(index=False).encode("utf-8-sig"),
            file_name="loops_summary.csv", mime="text/csv"
        )

        st.markdown("**상세표 (1 cycle step-by-step)**")
        if not df_steps.empty:
            runs = sorted(df_steps["run"].unique().tolist())
            pick_run = st.selectbox("Run 선택", runs, index=0, key="loop_run")
            loops_in_run = sorted(df_steps[df_steps["run"]==pick_run]["loop_id"].unique().tolist())
            pick_loop = st.selectbox("Loop 선택", loops_in_run, index=0, key="loop_id")

            view = df_steps[(df_steps["run"]==pick_run) & (df_steps["loop_id"]==pick_loop)] \
                    .sort_values("step_idx")
            st.dataframe(view, use_container_width=True)

            # 패턴 미니 요약(한 줄)
            lines = [f"({int(r.duration_s)}s) comment='{r.comment}' | actions='{r.actions}'"
                     for r in view.itertuples()]
            st.code("\n".join(lines), language="text")

            st.download_button(
                "CSV 다운로드(Loop 상세-선택)",
                data=view.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"loop_steps_{pick_loop}.csv", mime="text/csv"
            )
        else:
            st.info("업로드한 레시피에서 loop를 찾지 못했습니다.")

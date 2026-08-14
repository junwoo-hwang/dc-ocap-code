"""OCAP hold dashboard.

Left: hold list (dc_uly + dc_sol + dc_tts combined), newest first,
single-row selectable.
Right (top 2/3): scatter of the selected hold's item across its trend
dataframe (uly_trend / sol_trend / tts_trend), with UCL/LCL (blue) and
USL/LSL (red) reference lines. The held wafer is highlighted red
("{root_lot_id} #{wafer_no}" legend entry); same-lot points are a
darker gray than other lots.
Right (bottom 1/3): a comment box for the selected hold, saved to disk.

======================================================================
DATA PREP (mock — stands in for the real datalake pull, which can't be
shared here). Produces the final dc_uly/dc_sol/dc_tts and
uly_trend/sol_trend/tts_trend dataframes that the real pipeline already
has ready. Everything from the "여기부터 streamlit" marker down is the
actual dashboard and doesn't need to change when this section is
swapped for the real data pull.
======================================================================
"""

import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

LOT_ID_CHARS = list(string.ascii_uppercase + string.digits)

LINE_IDS = ["M14", "M16", "L1", "L2"]
HOLD_REASONS = [
    "OOC (Out of Control)",
    "SPEC OUT (USL Exceed)",
    "SPEC OUT (LSL Exceed)",
    "EQP ALARM",
    "TREND WARNING (7 POINT RUN)",
    "SUDDEN SHIFT",
    "MEASUREMENT DELAY",
]

PROBE_CARD_IDS = [f"PC{n:03d}" for n in range(1, 9)]
EQP_IDS = [f"PRB{n:02d}" for n in range(1, 7)]
LOT_TYPES = ["MP", "ENG", "MONITOR", "RND"]
RW_CNT_VALUES = [0, 1, 2, 3, 4, 5]

# product is identified by process_id: KNNU=uly, KNJO=sol, KNIK=tts
PRODUCT_CONFIG = {
    "ULY": {"n_items": 18, "center": 50, "spread": 6, "seed": 101, "process_id": "KNNU"},
    "SOL": {"n_items": 15, "center": 8, "spread": 1.5, "seed": 103, "process_id": "KNJO"},
    "TTS": {"n_items": 22, "center": 120, "spread": 10, "seed": 102, "process_id": "KNIK"},
}


def _random_datetime(rng: np.random.Generator, start: datetime, end: datetime, size: int) -> pd.Series:
    delta_seconds = int((end - start).total_seconds())
    offsets = rng.integers(0, delta_seconds, size=size)
    return pd.Series([start + timedelta(seconds=int(s)) for s in offsets])


def generate_probe_df(product: str, n_rows: int = 300) -> pd.DataFrame:
    """Generate a mock wide-format probe test ("trend") dataframe for a product.

    Columns: root_lot_id, wafer_id, tkout_time, probe_card_id, eqp_id,
    lot_type, rw_cnt, item1..itemN (N = PRODUCT_CONFIG[product]['n_items']).

    rw_cnt is the retest sequence number: for a given (root_lot_id, wafer_id)
    group sorted by tkout_time, the first row is rw_cnt=0, and each
    subsequent row (an actual retest of that same wafer, at a later
    tkout_time) increments it by 1. A wafer only appears more than once
    when it was genuinely retested. Retests (rw_cnt >= 1) don't re-measure
    every item, so a random subset of item columns is left as NaN on those
    rows; rw_cnt=0 rows always have every item filled.
    """
    cfg = PRODUCT_CONFIG[product]
    rng = np.random.default_rng(cfg["seed"])
    n_items = cfg["n_items"]

    # ~19% chance a wafer gets one more retest than its previous test;
    # this reproduces the ~80/15/3/1.5/0.4/0.1% split across rw_cnt 0-5
    CONTINUE_PROB = 0.19
    EXPECTED_CHAIN_LEN = 1.234  # sum_{k=0..5} CONTINUE_PROB**k
    n_base = max(1, round(n_rows / EXPECTED_CHAIN_LEN))

    n_lots = max(1, n_base // 5)
    root_lot_ids = set()
    while len(root_lot_ids) < n_lots:
        root_lot_ids.add("".join(rng.choice(LOT_ID_CHARS, size=5)))
    root_lot_ids = list(root_lot_ids)
    wafer_pool = [(lot, w) for lot in root_lot_ids for w in range(1, 26)]
    rng.shuffle(wafer_pool)
    base_wafers = wafer_pool[: min(n_base, len(wafer_pool))]

    start_dt = datetime(2026, 7, 1)
    end_dt = datetime(2026, 8, 12, 23, 59, 59)
    span_seconds = int((end_dt - start_dt).total_seconds())

    rows = []
    for lot, wafer_id in base_wafers:
        tkout_time = start_dt + timedelta(seconds=int(rng.integers(0, span_seconds)))

        chain_len = 1
        while chain_len < len(RW_CNT_VALUES) and rng.random() < CONTINUE_PROB:
            chain_len += 1

        for rw in range(chain_len):
            if rw > 0:
                tkout_time += timedelta(hours=int(rng.integers(2, 72)))

            if rw == 0:
                values = rng.normal(loc=cfg["center"], scale=cfg["spread"], size=n_items)
            else:
                values = np.full(n_items, np.nan)
                n_measured = rng.integers(1, n_items // 2 + 2)
                measured_cols = rng.choice(n_items, size=n_measured, replace=False)
                values[measured_cols] = rng.normal(loc=cfg["center"], scale=cfg["spread"], size=n_measured)

            row = {
                "root_lot_id": lot,
                "wafer_id": wafer_id,
                "tkout_time": tkout_time,
                "probe_card_id": rng.choice(PROBE_CARD_IDS),
                "eqp_id": rng.choice(EQP_IDS),
                "lot_type": rng.choice(LOT_TYPES, p=[0.7, 0.15, 0.1, 0.05]),
                "rw_cnt": rw,
            }
            for i in range(n_items):
                row[f"item{i + 1}"] = round(float(values[i]), 3) if not np.isnan(values[i]) else np.nan
            rows.append(row)

    return pd.DataFrame(rows).sort_values("tkout_time").reset_index(drop=True)


def generate_dc_for_product(product: str, trend_df: pd.DataFrame, n_rows: int = 50, seed: int | None = None) -> pd.DataFrame:
    """Generate a mock hold-event dataframe for a single product.

    Each hold event is tied to a real (root_lot_id, wafer_id, item column)
    combination sampled from `trend_df` (only rw_cnt=0 rows, since those
    have every item measured), so it can be looked up in the trend
    dataframe on the dashboard. process_id is fixed to the product's code.
    """
    rng = np.random.default_rng(seed)
    cfg = PRODUCT_CONFIG[product]
    base_rows = trend_df[trend_df["rw_cnt"] == 0]
    item_cols = [c for c in trend_df.columns if c.startswith("item")]

    rows = []
    for _ in range(n_rows):
        src = base_rows.iloc[rng.integers(0, len(base_rows))]
        item_id = rng.choice(item_cols)

        spread = cfg["spread"]
        base = rng.normal(loc=cfg["center"], scale=spread)
        usl = base + rng.uniform(spread * 1.5, spread * 2.5)
        lsl = base - rng.uniform(spread * 1.5, spread * 2.5)
        ucl = base + rng.uniform(spread * 0.6, spread * 1.2)
        lcl = base - rng.uniform(spread * 0.6, spread * 1.2)

        rows.append(
            {
                "root_lot_id": src["root_lot_id"],
                "wafer_id": src["wafer_id"],
                "item_id": item_id,
                "hold_inform": rng.choice(HOLD_REASONS),
                "ucl": round(ucl, 3),
                "lcl": round(lcl, 3),
                "usl": round(usl, 3),
                "lsl": round(lsl, 3),
                "step_seq": int(rng.integers(10, 500)),
                "line_id": rng.choice(LINE_IDS),
                "process_id": cfg["process_id"],
                "sub_item_id": f"{item_id}_{rng.integers(1, 4)}",
            }
        )

    dc_df = pd.DataFrame(rows)
    dc_df["hold_time"] = _random_datetime(
        rng, datetime(2026, 7, 1), datetime(2026, 8, 14, 23, 59, 59), n_rows
    ).sort_values().reset_index(drop=True)

    return dc_df[
        [
            "root_lot_id",
            "wafer_id",
            "hold_time",
            "item_id",
            "hold_inform",
            "ucl",
            "lcl",
            "usl",
            "lsl",
            "step_seq",
            "line_id",
            "process_id",
            "sub_item_id",
        ]
    ]


# per-product trend ("uly_trend" style naming, matches the real dataframes)
uly_trend = generate_probe_df("ULY")
sol_trend = generate_probe_df("SOL")
tts_trend = generate_probe_df("TTS")

TREND_FRAMES = {"ULY": uly_trend, "SOL": sol_trend, "TTS": tts_trend}

# per-product hold events ("dc_uly" style naming, matches the real dataframes)
dc_uly = generate_dc_for_product("ULY", uly_trend, n_rows=50, seed=201)
dc_sol = generate_dc_for_product("SOL", sol_trend, n_rows=50, seed=202)
dc_tts = generate_dc_for_product("TTS", tts_trend, n_rows=50, seed=203)


# ======================================================================
# 여기부터 streamlit
# ======================================================================

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="OCAP Hold Dashboard", layout="wide")

LEGEND_FIELD_OPTIONS = {
    "없음 (기본)": None,
    "probe_card_id": "probe_card_id",
    "eqp_id": "eqp_id",
    "lot_type": "lot_type",
    "rw_cnt": "rw_cnt",
}

HOVER_COLS = ["root_lot_id", "wafer_id", "probe_card_id", "eqp_id", "lot_type", "rw_cnt"]
HOVER_TEMPLATE = (
    "root_lot_id=%{customdata[0]}<br>"
    "wafer_id=%{customdata[1]}<br>"
    "probe_card_id=%{customdata[2]}<br>"
    "eqp_id=%{customdata[3]}<br>"
    "lot_type=%{customdata[4]}<br>"
    "rw_cnt=%{customdata[5]}<extra></extra>"
)

CATEGORY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
    "#17becf", "#bcbd22", "#7f7f7f", "#e377c2", "#aec7e8",
]

COMMENTS_PATH = Path(__file__).parent / "comments.csv"
COMMENT_COLS = ["root_lot_id", "wafer_id", "item_id", "comment", "status", "saved_at"]
STATUS_OPTIONS = ["Flow", "Retest", "Hold"]


def find_trend_df(root_lot_id: str, item_id: str):
    for product, tdf in TREND_FRAMES.items():
        if item_id in tdf.columns and (tdf["root_lot_id"] == root_lot_id).any():
            return product, tdf
    return None, None


def build_scatter(trend_df, item_id: str, bad_root_lot_id: str, bad_wafer_id: int,
                   legend_field: str | None, ucl: float, lcl: float, usl: float, lsl: float,
                   chart_height: int) -> go.Figure:
    plot_df = trend_df.dropna(subset=[item_id])
    is_bad = (plot_df["root_lot_id"] == bad_root_lot_id) & (plot_df["wafer_id"] == bad_wafer_id)
    others = plot_df[~is_bad]
    bad = plot_df[is_bad]
    bad_label = f"{bad_root_lot_id} #{bad_wafer_id}"

    fig = go.Figure()

    def add_group(grp: pd.DataFrame, color: str, name: str) -> None:
        if grp.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=grp["tkout_time"], y=grp[item_id],
                mode="markers", marker=dict(color=color, size=7),
                name=name,
                customdata=grp[HOVER_COLS].values,
                hovertemplate=HOVER_TEMPLATE,
            )
        )

    if legend_field is None:
        same_lot_mask = others["root_lot_id"] == bad_root_lot_id
        add_group(others[~same_lot_mask], "lightgray", "other")
        add_group(others[same_lot_mask], "dimgray", bad_root_lot_id)
    else:
        for i, (cat_val, grp) in enumerate(others.groupby(legend_field)):
            add_group(grp, CATEGORY_COLORS[i % len(CATEGORY_COLORS)], str(cat_val))

    if legend_field is None:
        fig.add_trace(
            go.Scatter(
                x=bad["tkout_time"], y=bad[item_id],
                mode="markers",
                marker=dict(color="red", size=11, line=dict(width=1, color="black")),
                name=bad_label,
                customdata=bad[HOVER_COLS].values,
                hovertemplate=HOVER_TEMPLATE,
            )
        )
    else:
        for cat_val, grp in bad.groupby(legend_field):
            fig.add_trace(
                go.Scatter(
                    x=grp["tkout_time"], y=grp[item_id],
                    mode="markers",
                    marker=dict(color="red", size=11, line=dict(width=1, color="black")),
                    name=f"{bad_label}_{cat_val}",
                    customdata=grp[HOVER_COLS].values,
                    hovertemplate=HOVER_TEMPLATE,
                )
            )

    fig.add_hline(y=ucl, line=dict(color="blue", dash="dash"), annotation_text="UCL", annotation_position="top left")
    fig.add_hline(y=lcl, line=dict(color="blue", dash="dash"), annotation_text="LCL", annotation_position="bottom left")
    fig.add_hline(y=usl, line=dict(color="red", dash="dash"), annotation_text="USL", annotation_position="top left")
    fig.add_hline(y=lsl, line=dict(color="red", dash="dash"), annotation_text="LSL", annotation_position="bottom left")

    fig.update_layout(
        xaxis_title="tkout_time",
        yaxis_title=item_id,
        legend_title=legend_field or "Legend",
        height=chart_height,
        margin=dict(t=30, b=30),
    )
    return fig


def load_comments() -> pd.DataFrame:
    if COMMENTS_PATH.exists():
        return pd.read_csv(COMMENTS_PATH)
    return pd.DataFrame(columns=COMMENT_COLS)


def save_comment(root_lot_id: str, wafer_id: str, item_id: str, comment_text: str, status: str) -> None:
    comments_df = load_comments()
    mask = (
        (comments_df["root_lot_id"] == root_lot_id)
        & (comments_df["wafer_id"] == wafer_id)
        & (comments_df["item_id"] == item_id)
    )
    now = datetime.now().isoformat(timespec="seconds")
    if mask.any():
        comments_df.loc[mask, "comment"] = comment_text
        comments_df.loc[mask, "status"] = status
        comments_df.loc[mask, "saved_at"] = now
    else:
        new_row = pd.DataFrame([{
            "root_lot_id": root_lot_id, "wafer_id": wafer_id, "item_id": item_id,
            "comment": comment_text, "status": status, "saved_at": now,
        }])
        comments_df = pd.concat([comments_df, new_row], ignore_index=True)
    comments_df.to_csv(COMMENTS_PATH, index=False)


st.title("Hold 현황 대시보드")

PANEL_HEIGHT = 650
TREND_HEIGHT = round(PANEL_HEIGHT * 2 / 3)
COMMENT_HEIGHT = PANEL_HEIGHT - TREND_HEIGHT

left, right = st.columns([2, 3])

DC_COLS = [
    "root_lot_id", "wafer_id", "hold_time", "item_id", "hold_inform",
    "ucl", "lcl", "usl", "lsl", "step_seq", "line_id", "process_id", "sub_item_id",
]
display_cols = ["hold_time", "root_lot_id", "wafer_id", "item_id", "hold_inform", "line_id", "process_id"]
PRODUCT_DC = {"ULY": dc_uly, "SOL": dc_sol, "TTS": dc_tts}

with left:
    title_col, switch_col = st.columns([2, 2])
    with title_col:
        st.subheader("DC OCAP List")
    with switch_col:
        selected_product = st.segmented_control(
            "제품", list(PRODUCT_DC.keys()), default="ULY", label_visibility="collapsed", key="product_switch"
        )
    selected_product = selected_product or "ULY"

    dc_df = PRODUCT_DC[selected_product]
    if dc_df is None or dc_df.empty or "hold_time" not in dc_df.columns:
        dc_sorted = pd.DataFrame(columns=DC_COLS)
        st.caption(f"{selected_product}: 현재 hold 건이 없습니다.")
    else:
        dc_sorted = dc_df.sort_values("hold_time", ascending=False).reset_index(drop=True)

    event = st.dataframe(
        dc_sorted[display_cols],
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=PANEL_HEIGHT,
        key=f"dc_table_{selected_product}",
    )
    selected_rows = event.selection.rows if event and event.selection else []

with right:
    st.subheader("Item Trend")
    sel = dc_sorted.iloc[selected_rows[0]] if selected_rows else None
    product, tdf = find_trend_df(sel["root_lot_id"], sel["item_id"]) if sel is not None else (None, None)

    with st.container(height=TREND_HEIGHT, border=True):
        if sel is None:
            st.info("왼쪽에서 hold 행을 클릭하면 trend 차트가 표시됩니다.")
        elif tdf is None:
            st.warning("매칭되는 trend 데이터를 찾지 못했습니다.")
        else:
            legend_spacer, legend_col = st.columns([4, 1])
            with legend_col:
                legend_label = st.selectbox("Legend", list(LEGEND_FIELD_OPTIONS.keys()), index=0)
            legend_field = LEGEND_FIELD_OPTIONS[legend_label]

            fig = build_scatter(
                tdf, sel["item_id"], sel["root_lot_id"], sel["wafer_id"], legend_field,
                sel["ucl"], sel["lcl"], sel["usl"], sel["lsl"],
                chart_height=TREND_HEIGHT - 190,
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"제품: {product} · root_lot_id: {sel['root_lot_id']} · "
                f"wafer_id: {sel['wafer_id']} · item: {sel['item_id']}"
            )

    with st.container(height=COMMENT_HEIGHT, border=True):
        if sel is None or tdf is None:
            st.caption("Comment")
        else:
            comments_df = load_comments()
            mask = (
                (comments_df["root_lot_id"] == sel["root_lot_id"])
                & (comments_df["wafer_id"] == sel["wafer_id"])
                & (comments_df["item_id"] == sel["item_id"])
            )
            existing_comment = comments_df.loc[mask, "comment"].iloc[-1] if mask.any() else ""
            existing_status = comments_df.loc[mask, "status"].iloc[-1] if mask.any() else None
            existing_status = existing_status if existing_status in STATUS_OPTIONS else None
            comment_key = f"{sel['root_lot_id']}_{sel['wafer_id']}_{sel['item_id']}"

            comment_text = st.text_area(
                "Comment", value=existing_comment, height=COMMENT_HEIGHT - 110, key=f"comment_input_{comment_key}"
            )
            row_spacer, row_status, row_save = st.columns([3, 3, 1])
            with row_status:
                status_choice = st.segmented_control(
                    "Status", STATUS_OPTIONS, default=existing_status,
                    label_visibility="collapsed", key=f"status_input_{comment_key}",
                )
            with row_save:
                save_clicked = st.button("저장", key=f"save_btn_{comment_key}")
            if save_clicked:
                if status_choice is None:
                    st.error("Flow / Retest / Hold 중 하나를 선택해주세요.")
                else:
                    save_comment(sel["root_lot_id"], sel["wafer_id"], sel["item_id"], comment_text, status_choice)
                    st.success("저장되었습니다.")

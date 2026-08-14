"""OCAP hold dashboard.

Left: dc (hold events), newest first, single-row selectable.
Right (top 2/3): scatter of the selected hold's item across its trend
dataframe (uly/tts/sol), with UCL/LCL (blue) and USL/LSL (red) reference
lines. The held wafer is highlighted red ("{root_lot_id} #{wafer_no}"
legend entry); same-lot points are a darker gray than other lots.
Right (bottom 1/3): a comment box for the selected hold, saved to disk.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mock_data import TREND_FRAMES, dc

st.set_page_config(page_title="OCAP Hold Dashboard", layout="wide")

LEGEND_FIELD_OPTIONS = {
    "없음 (기본)": None,
    "probe_card_id": "probe_card_id",
    "eqp_id": "eqp_id",
    "lot_type": "lot_type",
    "rw_cnt": "rw_cnt",
}

HOVER_COLS = ["wafer_id", "probe_card_id", "eqp_id", "lot_type", "rw_cnt"]
HOVER_TEMPLATE = (
    "wafer_id=%{customdata[0]}<br>"
    "probe_card_id=%{customdata[1]}<br>"
    "eqp_id=%{customdata[2]}<br>"
    "lot_type=%{customdata[3]}<br>"
    "rw_cnt=%{customdata[4]}<extra></extra>"
)

COMMENTS_PATH = Path(__file__).parent / "comments.csv"
COMMENT_COLS = ["root_lot_id", "wafer_id", "item_id", "comment", "saved_at"]


def find_trend_df(root_lot_id: str, item_id: str):
    for product, tdf in TREND_FRAMES.items():
        if item_id in tdf.columns and (tdf["root_lot_id"] == root_lot_id).any():
            return product, tdf
    return None, None


def wafer_no(wafer_id: str) -> int:
    return int(wafer_id.split(".")[-1])


def build_scatter(trend_df, item_id: str, bad_root_lot_id: str, bad_wafer_id: str,
                   legend_field: str | None, ucl: float, lcl: float, usl: float, lsl: float) -> go.Figure:
    plot_df = trend_df.dropna(subset=[item_id])
    others = plot_df[plot_df["wafer_id"] != bad_wafer_id]
    bad = plot_df[plot_df["wafer_id"] == bad_wafer_id]
    bad_label = f"{bad_root_lot_id} #{wafer_no(bad_wafer_id)}"

    fig = go.Figure()

    if legend_field is None:
        colors = ["dimgray" if lot == bad_root_lot_id else "lightgray" for lot in others["root_lot_id"]]
        fig.add_trace(
            go.Scatter(
                x=others["tkout_time"], y=others[item_id],
                mode="markers", marker=dict(color=colors, size=7),
                name="other",
                customdata=others[HOVER_COLS].values,
                hovertemplate=HOVER_TEMPLATE,
            )
        )
    else:
        for cat_val, grp in others.groupby(legend_field):
            fig.add_trace(
                go.Scatter(
                    x=grp["tkout_time"], y=grp[item_id],
                    mode="markers", marker=dict(color="gray", size=7),
                    name=f"{legend_field}={cat_val}",
                    customdata=grp[HOVER_COLS].values,
                    hovertemplate=HOVER_TEMPLATE,
                )
            )

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

    fig.add_hline(y=ucl, line=dict(color="blue", dash="dash"), annotation_text="UCL", annotation_position="top left")
    fig.add_hline(y=lcl, line=dict(color="blue", dash="dash"), annotation_text="LCL", annotation_position="bottom left")
    fig.add_hline(y=usl, line=dict(color="red", dash="dash"), annotation_text="USL", annotation_position="top left")
    fig.add_hline(y=lsl, line=dict(color="red", dash="dash"), annotation_text="LSL", annotation_position="bottom left")

    fig.update_layout(
        xaxis_title="tkout_time",
        yaxis_title=item_id,
        legend_title=legend_field or "Legend",
        height=430,
        margin=dict(t=40),
    )
    return fig


def load_comments() -> pd.DataFrame:
    if COMMENTS_PATH.exists():
        return pd.read_csv(COMMENTS_PATH)
    return pd.DataFrame(columns=COMMENT_COLS)


def save_comment(root_lot_id: str, wafer_id: str, item_id: str, comment_text: str) -> None:
    comments_df = load_comments()
    mask = (
        (comments_df["root_lot_id"] == root_lot_id)
        & (comments_df["wafer_id"] == wafer_id)
        & (comments_df["item_id"] == item_id)
    )
    now = datetime.now().isoformat(timespec="seconds")
    if mask.any():
        comments_df.loc[mask, "comment"] = comment_text
        comments_df.loc[mask, "saved_at"] = now
    else:
        new_row = pd.DataFrame([{
            "root_lot_id": root_lot_id, "wafer_id": wafer_id, "item_id": item_id,
            "comment": comment_text, "saved_at": now,
        }])
        comments_df = pd.concat([comments_df, new_row], ignore_index=True)
    comments_df.to_csv(COMMENTS_PATH, index=False)


st.title("Hold 현황 대시보드")

left, right = st.columns([2, 3])

dc_sorted = dc.sort_values("hold_time", ascending=False).reset_index(drop=True)
display_cols = ["hold_time", "root_lot_id", "wafer_id", "item_id", "hold_inform", "line_id", "process_id"]

with left:
    st.subheader("Hold 리스트 (최근순)")
    event = st.dataframe(
        dc_sorted[display_cols],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=650,
        key="dc_table",
    )
    selected_rows = event.selection.rows if event and event.selection else []

with right:
    st.subheader("Item Trend")
    if not selected_rows:
        st.info("왼쪽에서 hold 행을 클릭하면 trend 차트가 표시됩니다.")
    else:
        sel = dc_sorted.iloc[selected_rows[0]]
        product, tdf = find_trend_df(sel["root_lot_id"], sel["item_id"])

        if tdf is None:
            st.warning("매칭되는 trend 데이터를 찾지 못했습니다.")
        else:
            legend_label = st.selectbox("Legend 기준", list(LEGEND_FIELD_OPTIONS.keys()), index=0)
            legend_field = LEGEND_FIELD_OPTIONS[legend_label]

            fig = build_scatter(
                tdf, sel["item_id"], sel["root_lot_id"], sel["wafer_id"], legend_field,
                sel["ucl"], sel["lcl"], sel["usl"], sel["lsl"],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"제품: {product} · root_lot_id: {sel['root_lot_id']} · "
                f"wafer_id: {sel['wafer_id']} · item: {sel['item_id']}"
            )

            st.markdown("---")
            comments_df = load_comments()
            mask = (
                (comments_df["root_lot_id"] == sel["root_lot_id"])
                & (comments_df["wafer_id"] == sel["wafer_id"])
                & (comments_df["item_id"] == sel["item_id"])
            )
            existing_comment = comments_df.loc[mask, "comment"].iloc[-1] if mask.any() else ""
            comment_key = f"{sel['root_lot_id']}_{sel['wafer_id']}_{sel['item_id']}"

            comment_text = st.text_area(
                "Comment", value=existing_comment, height=110, key=f"comment_input_{comment_key}"
            )
            if st.button("저장", key=f"save_btn_{comment_key}"):
                save_comment(sel["root_lot_id"], sel["wafer_id"], sel["item_id"], comment_text)
                st.success("저장되었습니다.")

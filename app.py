"""OCAP hold dashboard.

Left: dc (hold events), newest first, single-row selectable.
Right: scatter of the selected hold's item across its trend dataframe
(uly/tts/sol), with UCL/LCL (blue) and USL/LSL (red) reference lines,
and the held wafer highlighted in red as a toggleable "bad" legend entry.
"""

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


def find_trend_df(root_lot_id: str, item_id: str):
    for product, tdf in TREND_FRAMES.items():
        if item_id in tdf.columns and (tdf["root_lot_id"] == root_lot_id).any():
            return product, tdf
    return None, None


def build_scatter(trend_df, item_id: str, bad_wafer_id: str, legend_field: str | None,
                   ucl: float, lcl: float, usl: float, lsl: float) -> go.Figure:
    plot_df = trend_df.dropna(subset=[item_id])
    others = plot_df[plot_df["wafer_id"] != bad_wafer_id]
    bad = plot_df[plot_df["wafer_id"] == bad_wafer_id]

    fig = go.Figure()

    if legend_field is None:
        fig.add_trace(
            go.Scatter(
                x=others["tkout_time"], y=others[item_id],
                mode="markers", marker=dict(color="gray", size=7),
                name="normal",
            )
        )
    else:
        for cat_val, grp in others.groupby(legend_field):
            fig.add_trace(
                go.Scatter(
                    x=grp["tkout_time"], y=grp[item_id],
                    mode="markers", marker=dict(color="gray", size=7),
                    name=f"{legend_field}={cat_val}",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=bad["tkout_time"], y=bad[item_id],
            mode="markers",
            marker=dict(color="red", size=11, line=dict(width=1, color="black")),
            name="bad",
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
        height=600,
        margin=dict(t=40),
    )
    return fig


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
                tdf, sel["item_id"], sel["wafer_id"], legend_field,
                sel["ucl"], sel["lcl"], sel["usl"], sel["lsl"],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"제품: {product} · root_lot_id: {sel['root_lot_id']} · "
                f"wafer_id: {sel['wafer_id']} · item: {sel['item_id']}"
            )

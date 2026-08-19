"""Generate a self-contained dc_ocap.html snapshot (no server needed to view it).

Run this on a schedule (hourly, from wherever pull_data() can actually reach
the company system -- the internal VSCode environment, not the portal
server) and upload the resulting file next to the portal's other static
reports (e.g. S3), the same way solomon_eds_templete.html etc. get there.
This script only writes the local file; wiring up that upload step is left
to the scheduler, since it depends on credentials this repo doesn't have.

Reuses pull_data()/check_data() from app.py, so swapping pull_data()'s body
for the real company-system pull benefits both this export path and the
Streamlit portal page (show_dc_ocap()) from a single change.

The template (dc_ocap_template.html) re-implements show_dc_ocap()'s whole
interaction model in vanilla JS + Plotly.js -- see the comment at the top
of that file for why a straight "HTML export" of the Streamlit page isn't
possible and this had to be a hand port instead.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.offline as pyo

from app import check_data, pull_data

KST = timezone(timedelta(hours=9))
HERE = Path(__file__).parent
TEMPLATE_PATH = HERE / "dc_ocap_template.html"
OUTPUT_PATH = HERE / "dc_ocap.html"

# kept in sync with dc_ocap_template.html's META_TREND_COLS -- everything
# else in a trend row is an item measurement column
META_TREND_COLS = ["root_lot_id", "wafer_id", "tkout_time", "probe_card_id", "eqp_id", "lot_type", "rw_cnt"]


def _clean(value):
    """One cell -> a JSON-safe value: NaN/NaT -> None, Timestamp -> ISO
    string, numpy scalar -> native Python (json.dumps chokes on numpy
    int64/float64)."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def build() -> Path:
    dc_uly, dc_sol, dc_tts, uly_trend, sol_trend, tts_trend = pull_data()

    product_dc = {"ULY": dc_uly, "SOL": dc_sol, "TTS": dc_tts}
    product_trend = {"ULY": uly_trend, "SOL": sol_trend, "TTS": tts_trend}

    # same validation the Streamlit page runs before trusting the data --
    # a bad schema should fail the scheduled build loudly rather than ship
    # a broken dc_ocap.html
    problems = check_data(product_dc, product_trend)
    if problems:
        raise SystemExit(
            "pull_data() 가 돌려준 데이터가 대시보드 형식과 맞지 않습니다:\n"
            + "\n".join(f"- {p}" for p in problems)
        )

    data = {
        "dc": {p: _records(product_dc[p]) for p in product_dc},
        "trend": {p: _records(product_trend[p]) for p in product_trend},
        "itemCols": {
            p: [c for c in product_trend[p].columns if c not in META_TREND_COLS]
            for p in product_trend
        },
    }

    generated_at = datetime.now(KST).strftime("%y/%m/%d %H:%M")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = (
        template
        .replace("__GENERATED_AT__", generated_at)
        .replace("/*__DATA_JSON__*/null", json.dumps(data, ensure_ascii=False))
        # embedded rather than loaded from the public CDN: the portal server
        # or its viewers may not have outbound internet access, only
        # reachability to wherever this file itself gets hosted
        .replace("/*__PLOTLY_JS__*/", pyo.offline.get_plotlyjs())
    )
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")
    return OUTPUT_PATH


if __name__ == "__main__":
    build()

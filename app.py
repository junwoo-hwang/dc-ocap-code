"""OCAP hold dashboard.

Left: the selected product's hold list (dc_uly / dc_sol / dc_tts),
grouped to one row per lot_id, newest first, single-row selectable,
with a ULY/SOL/TTS switch.
Right (top 2/3): scatter of one measurement item across that product's
trend dataframe (uly_trend / sol_trend / tts_trend), with UCL/LCL
(blue) and USL/LSL (red) reference lines. Every wafer held for that
item is red under a single legend entry; with no legend field chosen,
other wafers from the same lot are a darker gray than the rest. The
arrows step through the lot's items one at a time.
Right (bottom 1/3): the disposition recorded in the company system and
merged into dc -- comment, then owner and code -- shown read-only.

======================================================================
DATA PREP (mock — stands in for the real pull, which can't be shared
here). Replace this whole section with the real company-system pull;
it only has to end up with pull_data() returning these six dataframes:
dc_uly / dc_sol / dc_tts and uly_trend / sol_trend / tts_trend.

Keep the pull inside pull_data() rather than at module level: Streamlit
re-runs this file top to bottom on every click, so module-level code
would re-query on every row selection. The dashboard calls it through
@st.cache_data.

This section deliberately uses no streamlit -- everything from the
"여기부터 streamlit" marker down is self-contained (its own imports
included), so replacing this section can't break the dashboard.
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

# merged in from the company system alongside owner/code/comment
OWNER_DEPTS = ["품질기술팀", "공정기술팀", "설비기술팀", "수율개선팀"]
OWNER_NAMES = ["김민준", "이서연", "박지훈", "최수빈", "정하늘", "강도윤"]
CODES = ["Flow", "Retest", "Hold"]
COMMENTS_BY_CODE = {
    "Flow": [
        "재측정 결과 규격 내 확인, 진행 조치함",
        "설비 계측 오차로 판단됨. 후속 lot 정상 확인되어 flow 처리",
        "single point 이탈이며 경향성 없음. 진행",
    ],
    "Retest": [
        "측정값 이상으로 재측정 요청",
        "probe card 접촉 불량 의심되어 재측정 진행",
        "동일 조건 재측정 후 재판정 예정",
    ],
    "Hold": [
        "규격 이탈 확인. 공정팀 원인 분석 요청",
        "연속 이탈 경향 확인되어 hold 유지",
        "설비 이상 이력과 연계 확인 필요. hold 유지",
    ],
}

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
    # a set of strings iterates in a process-dependent order, which would
    # make the "seeded" mock differ on every run; sort to pin it down
    root_lot_ids = set()
    while len(root_lot_ids) < n_lots:
        root_lot_ids.add("".join(rng.choice(LOT_ID_CHARS, size=5)))
    root_lot_ids = sorted(root_lot_ids)
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
    combination sampled from `trend_df`, so it can be looked up in the
    trend dataframe on the dashboard. process_id is fixed to the product's
    code.

    rw_cnt says which measurement of that wafer was held (0 = the first
    test, 1 = the first retest, ...). One hold record carries one comment,
    so it takes part in the key the company system's comment table joins
    on -- the same wafer/item held again after a retest is a separate
    record with its own comment.
    """
    rng = np.random.default_rng(seed)
    cfg = PRODUCT_CONFIG[product]
    base_rows = trend_df[trend_df["rw_cnt"] == 0]
    item_cols = [c for c in trend_df.columns if c.startswith("item")]

    rows = []
    # holds arrive as lots, not as single wafers: one lot_id covers several
    # wafers and often several measurement items, which is what the
    # dashboard groups on
    while len(rows) < n_rows:
        src_lot = base_rows.iloc[rng.integers(0, len(base_rows))]["root_lot_id"]
        lot_rows = base_rows[base_rows["root_lot_id"] == src_lot]
        lot_id = f"{src_lot}.{rng.integers(1, 9)}"
        hold_time = _random_datetime(rng, datetime(2026, 7, 1), datetime(2026, 8, 14, 23, 59, 59), 1).iloc[0]

        n_waf = min(int(rng.integers(1, 6)), len(lot_rows))
        wafers = rng.choice(lot_rows["wafer_id"].unique(), size=min(n_waf, lot_rows["wafer_id"].nunique()), replace=False)
        items = rng.choice(item_cols, size=int(rng.integers(1, 4)), replace=False)

        owner = f"{rng.choice(OWNER_DEPTS)} {rng.choice(OWNER_NAMES)}"
        code = rng.choice(CODES, p=[0.55, 0.3, 0.15])
        comment = rng.choice(COMMENTS_BY_CODE[code])

        for item_id in items:
            spread = cfg["spread"]
            base = rng.normal(loc=cfg["center"], scale=spread)
            usl = base + rng.uniform(spread * 1.5, spread * 2.5)
            lsl = base - rng.uniform(spread * 1.5, spread * 2.5)
            ucl = base + rng.uniform(spread * 0.6, spread * 1.2)
            lcl = base - rng.uniform(spread * 0.6, spread * 1.2)
            hold_inform = rng.choice(HOLD_REASONS)
            step_seq = int(rng.integers(10, 500))
            line_id = rng.choice(LINE_IDS)

            for wafer_id in wafers:
                # point at a measurement that actually exists for this item:
                # retests only re-measure some items, so not every rw_cnt of
                # a wafer has a value for the item being held
                measured = trend_df[
                    (trend_df["root_lot_id"] == src_lot)
                    & (trend_df["wafer_id"] == wafer_id)
                    & trend_df[item_id].notna()
                ]
                rw_cnt = int(measured["rw_cnt"].iloc[rng.integers(0, len(measured))]) if len(measured) else 0

                rows.append(
                    {
                        "lot_id": lot_id,
                        "root_lot_id": src_lot,
                        "wafer_id": wafer_id,
                        "rw_cnt": rw_cnt,
                        "hold_time": hold_time,
                        "item_id": item_id,
                        "hold_inform": hold_inform,
                        "ucl": round(ucl, 3),
                        "lcl": round(lcl, 3),
                        "usl": round(usl, 3),
                        "lsl": round(lsl, 3),
                        "step_seq": step_seq,
                        "line_id": line_id,
                        "process_id": cfg["process_id"],
                        "sub_item_id": f"{item_id}_{rng.integers(1, 4)}",
                        # merged in from the company system, where the
                        # disposition is actually recorded
                        "owner": owner,
                        "code": code,
                        "comment": comment,
                    }
                )

    return pd.DataFrame(rows)[
        [
            "lot_id",
            "root_lot_id",
            "wafer_id",
            "rw_cnt",
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
            "owner",
            "code",
            "comment",
        ]
    ]


def pull_data():
    """Return (dc_uly, dc_sol, dc_tts, uly_trend, sol_trend, tts_trend).

    Put the real company-system pull in here. It must be a function, not
    bare module-level code: Streamlit re-runs this file top to bottom on
    every click, so anything at module level would be re-fetched on every
    row selection. The dashboard below calls this through a cache.
    """
    uly_trend = generate_probe_df("ULY")
    sol_trend = generate_probe_df("SOL")
    tts_trend = generate_probe_df("TTS")

    dc_uly = generate_dc_for_product("ULY", uly_trend, n_rows=50, seed=201)
    dc_sol = generate_dc_for_product("SOL", sol_trend, n_rows=50, seed=202)
    dc_tts = generate_dc_for_product("TTS", tts_trend, n_rows=50, seed=203)

    return dc_uly, dc_sol, dc_tts, uly_trend, sol_trend, tts_trend


# ======================================================================
# 여기부터 streamlit
# ======================================================================

# imported here rather than at the top of the file so this section keeps
# working if the data prep above is replaced wholesale
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# must be the first streamlit call in the script
st.set_page_config(page_title="OCAP Hold Dashboard", layout="wide")

# KST is pinned at UTC+9 rather than read from the host clock, so the
# header timestamp stays correct wherever the app is deployed.
KST = timezone(timedelta(hours=9))


# columns the dashboard below reads; anything missing would otherwise
# surface as a KeyError deep in a callback
DC_REQUIRED = [
    "lot_id", "root_lot_id", "wafer_id", "rw_cnt", "hold_time", "item_id",
    "hold_inform", "ucl", "lcl", "usl", "lsl", "line_id", "process_id",
    # merged in from the company system, where the disposition is recorded
    "owner", "code", "comment",
]
TREND_REQUIRED = [
    "root_lot_id", "wafer_id", "tkout_time",
    "probe_card_id", "eqp_id", "lot_type", "rw_cnt",
]
# the grouped hold list shown on the left, one row per lot_id
GROUP_COLS = ["hold_time", "lot_id", "wafer_id", "item", "hold_inform", "code", "owner"]


def sort_wafers(values) -> list:
    """Wafer numbers in numeric order, with any non-numeric ones last."""
    normed = {norm_wafer(v) for v in values}
    nums = sorted(v for v in normed if isinstance(v, int))
    rest = sorted(str(v) for v in normed if not isinstance(v, int))
    return nums + rest


def summarize(values) -> str:
    """'item1' for one distinct value, 'item1외 2건' for several."""
    seen = list(dict.fromkeys(str(v) for v in values))
    if not seen:
        return ""
    return seen[0] if len(seen) == 1 else f"{seen[0]}외 {len(seen) - 1}건"


def group_holds(dc_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the hold list to one row per lot_id.

    A single lot is held as one event covering several wafers and often
    several measurement items, so showing one row per (wafer, item) buries
    the engineer in near-duplicate rows. Wafers are listed out ("2,3,4")
    since the count is small and it says which wafers are affected; items
    are summarized ("item1외 2건") because the chart pages through them
    one at a time anyway.
    """
    if dc_df.empty or "lot_id" not in dc_df.columns:
        return pd.DataFrame(columns=GROUP_COLS)

    rows = []
    for lot_id, grp in dc_df.groupby("lot_id", dropna=False, sort=False):
        rows.append({
            "hold_time": grp["hold_time"].max(),
            "lot_id": lot_id,
            "wafer_id": ",".join(str(w) for w in sort_wafers(grp["wafer_id"])),
            "item": summarize(grp["item_id"]),
            "hold_inform": summarize(grp["hold_inform"]),
            "code": summarize(grp["code"]),
            "owner": summarize(grp["owner"]),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("hold_time", ascending=False)
        .reset_index(drop=True)
    )


def norm_lot(value) -> str:
    """Canonical root_lot_id for matching (stray whitespace removed)."""
    return str(value).strip()


def norm_wafer(value):
    """Canonical wafer number for matching.

    dc may store it zero-padded as text or category ("03") while the
    trend table has a plain int (3), so both are reduced to an int where
    possible and to trimmed text otherwise.
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return str(value).strip()


def to_float(value) -> float | None:
    """Best-effort float, e.g. for a BigQuery NUMERIC that came back as
    text or Decimal. None (not NaN) so callers can tell "absent" from
    "unparseable"; plotly's add_hline does arithmetic on y internally
    and raises a TypeError several frames deep if it gets a str."""
    try:
        v = float(value)
        return v if pd.notna(v) else None
    except (TypeError, ValueError):
        return None


def resolve_item_col(trend_df: pd.DataFrame, item_id) -> str | None:
    """Map a dc item_id onto the trend column holding that measurement.

    The two tables don't always agree on capitalisation (dc may say
    "ITEM3" where the trend column is "item3"), so an exact match is
    tried first and a case-insensitive one -- ignoring stray whitespace
    -- after. Returns None if nothing matches, or if the only matches
    are case-insensitive ones that are ambiguous between themselves.
    """
    if item_id in trend_df.columns:
        return item_id
    key = str(item_id).strip().lower()
    matches = [c for c in trend_df.columns if str(c).strip().lower() == key]
    return matches[0] if len(matches) == 1 else None


def check_data(product_dc: dict, trend_frames: dict) -> list[str]:
    """Return human-readable problems with what pull_data() handed back.

    Runs on the real data the first time it is plugged in, so a schema
    mismatch reads as a plain list of what to fix instead of a KeyError.
    Called from inside the cached load_data(), because scanning full
    trend tables on every rerun would cost more than the checks are
    worth (~0.8s per click on 200k rows).
    """
    problems = []
    for product in product_dc:
        dc_df, trend_df = product_dc[product], trend_frames[product]

        for label, df, required in (
            (f"dc_{product.lower()}", dc_df, DC_REQUIRED),
            (f"{product.lower()}_trend", trend_df, TREND_REQUIRED),
        ):
            if not isinstance(df, pd.DataFrame):
                problems.append(f"{label}: DataFrame 이 아닙니다 ({type(df).__name__}).")
                continue
            missing = [c for c in required if c not in df.columns]
            if missing:
                problems.append(f"{label}: 컬럼 없음 -> {', '.join(missing)}")

        if not isinstance(dc_df, pd.DataFrame) or not isinstance(trend_df, pd.DataFrame):
            continue

        # NAT_RATIO_LIMIT: pd.to_datetime(errors="coerce") turns anything it
        # can't parse into NaT instead of raising, so a wrong format string
        # or wrong source column silently turns the whole column into NaT
        # with no error at all. A handful of genuinely bad rows is a normal
        # data-quality reality and shouldn't block the dashboard, but most
        # of a column failing means the conversion itself is broken.
        NAT_RATIO_LIMIT = 0.5
        for label, df, col in (
            (f"dc_{product.lower()}", dc_df, "hold_time"),
            (f"{product.lower()}_trend", trend_df, "tkout_time"),
        ):
            if col not in df.columns:
                continue
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                problems.append(
                    f"{label}: {col} 이 날짜형이 아닙니다 ({df[col].dtype}). "
                    f"pd.to_datetime() 으로 변환하세요."
                )
            elif not df.empty and df[col].isna().mean() > NAT_RATIO_LIMIT:
                problems.append(
                    f"{label}: {col} 의 {df[col].isna().mean():.0%} 가 NaT 입니다 "
                    f"(원본 값 형식이 pd.to_datetime 으로 파싱되지 않는 것으로 보입니다)."
                )

        # ucl/lcl/usl/lsl must be numeric: plotly's add_hline raises a
        # TypeError deep inside its own layout code (not a clean KeyError)
        # if one of these came back as text (e.g. a BigQuery NUMERIC that
        # round-tripped as str/Decimal). Cheap to check in full -- only 4
        # columns, unlike the per-item trend checks below.
        for limit_col in ("ucl", "lcl", "usl", "lsl"):
            if limit_col not in dc_df.columns or dc_df.empty:
                continue
            bad_count = dc_df[limit_col].map(lambda v: to_float(v) is None and pd.notna(v)).sum()
            if bad_count:
                problems.append(
                    f"dc_{product.lower()}: {limit_col} 의 {bad_count}개 값이 숫자로 "
                    f"변환되지 않습니다 (dtype={dc_df[limit_col].dtype}). float 로 변환하세요."
                )

        # dc.item_id must name a real column in that product's trend
        # (case-insensitively -- resolve_item_col handles the difference)
        if "item_id" in dc_df.columns and not dc_df.empty:
            unknown = sorted(
                str(i) for i in set(dc_df["item_id"])
                if resolve_item_col(trend_df, i) is None
            )
            if unknown:
                problems.append(
                    f"dc_{product.lower()}: item_id {unknown[:5]} 이(가) "
                    f"{product.lower()}_trend 의 컬럼에 없습니다."
                )

        # a hold finds its wafer by the (root_lot_id, wafer_id) pair. The
        # dtypes may differ -- norm_lot/norm_wafer absorb that -- so what
        # matters is whether the normalized pairs actually line up.
        pair_cols = {"root_lot_id", "wafer_id"}
        if pair_cols <= set(dc_df.columns) and pair_cols <= set(trend_df.columns) and not dc_df.empty:
            trend_pairs = set(zip(trend_df["root_lot_id"].map(norm_lot),
                                  trend_df["wafer_id"].map(norm_wafer)))
            dc_pairs = set(zip(dc_df["root_lot_id"].map(norm_lot),
                               dc_df["wafer_id"].map(norm_wafer)))
            missing = dc_pairs - trend_pairs
            if len(missing) == len(dc_pairs):
                problems.append(
                    f"dc_{product.lower()}: (root_lot_id, wafer_id) 가 "
                    f"{product.lower()}_trend 에서 하나도 매칭되지 않습니다. "
                    f"dc 예시 {sorted(missing)[:3]}"
                )
            elif missing:
                problems.append(
                    f"dc_{product.lower()}: {len(missing)}/{len(dc_pairs)} 건의 "
                    f"(root_lot_id, wafer_id) 가 {product.lower()}_trend 에 없습니다 "
                    f"(해당 hold 는 빨간 점이 안 찍힘). 예시 {sorted(missing)[:3]}"
                )
    return problems


# Streamlit re-runs this file on every click, so pull_data() is called
# through a cache -- otherwise every row selection would re-query the
# company system. ttl is how stale the data may get before the next
# interaction refetches it; raise or lower it to taste, and use the app's
# ⋮ menu > Clear cache to force a refresh.
@st.cache_data(ttl=600, show_spinner="데이터 불러오는 중...")
def load_data():
    frames = pull_data()
    dc_uly, dc_sol, dc_tts, uly_trend, sol_trend, tts_trend = frames
    # checked here rather than on every rerun: it scans the whole trend
    # tables, which is far too slow to repeat on each click
    problems = check_data(
        {"ULY": dc_uly, "SOL": dc_sol, "TTS": dc_tts},
        {"ULY": uly_trend, "SOL": sol_trend, "TTS": tts_trend},
    )
    # stamped inside the cache, so the header reports when the data was
    # actually fetched rather than when the page was last re-rendered
    loaded_at = datetime.now(KST).strftime("%y/%m/%d %H:%M")
    return (*frames, loaded_at, problems)


dc_uly, dc_sol, dc_tts, uly_trend, sol_trend, tts_trend, DATA_LOADED_AT, _problems = load_data()

TREND_FRAMES = {"ULY": uly_trend, "SOL": sol_trend, "TTS": tts_trend}
PRODUCT_DC = {"ULY": dc_uly, "SOL": dc_sol, "TTS": dc_tts}

if _problems:
    st.error("pull_data() 가 돌려준 데이터가 대시보드 형식과 맞지 않습니다:")
    for _p in _problems:
        st.write("- " + _p)
    st.stop()

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

# red is reserved for the held wafer, so it is kept out of this palette.
# 20 entries so a high-cardinality field (many probe cards / eqp ids in the
# queried window) doesn't wrap onto a duplicate color too quickly.
CATEGORY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
    "#17becf", "#bcbd22", "#7f7f7f", "#e377c2", "#aec7e8",
    "#3182bd", "#fdae6b", "#74c476", "#9e9ac8", "#a1866f",
    "#6baed6", "#c7e9c0", "#525252", "#f7b6d2", "#dbdb8d",
]

def find_trend_df(product: str, item_id: str):
    """Return (trend dataframe, its column for item_id), or (None, None).

    The product is known from the hold list's own selection, so it is
    used directly instead of searching every trend frame for a matching
    root_lot_id -- a lot that appears under more than one product would
    otherwise chart the wrong product's data. The column is resolved
    rather than taken literally because dc and the trend tables can
    capitalise item ids differently.
    """
    tdf = TREND_FRAMES.get(product)
    if tdf is None or tdf.empty:
        return None, None
    item_col = resolve_item_col(tdf, item_id)
    if item_col is None:
        return None, None
    return tdf, item_col


def build_scatter(trend_df, item_id: str, bad_pairs: set, bad_label: str,
                   legend_field: str | None, ucl, lcl, usl, lsl,
                   chart_height: int) -> go.Figure:
    """Scatter one measurement item over time.

    `bad_pairs` is the set of normalized (root_lot_id, wafer_id) pairs
    held for this item -- a lot is held as a whole, so several wafers are
    highlighted together under one legend entry rather than one each.
    """
    ucl, lcl, usl, lsl = to_float(ucl), to_float(lcl), to_float(usl), to_float(lsl)

    # coerce before dropna: a numeric column that came back as text/object
    # (BigQuery NUMERIC/DECIMAL) would otherwise keep its non-null string
    # values and only fail once plotly tries to lay out the chart
    plot_df = trend_df.assign(**{item_id: pd.to_numeric(trend_df[item_id], errors="coerce")})
    plot_df = plot_df.dropna(subset=[item_id])
    # match on the pair (wafer numbers 1-25 repeat across lots), and
    # normalize both sides: dc and the trend table need not agree on how
    # a lot id is padded or whether the wafer number is text or an int
    row_pairs = list(zip(plot_df["root_lot_id"].map(norm_lot),
                         plot_df["wafer_id"].map(norm_wafer)))
    is_bad_row = pd.Series([p in bad_pairs for p in row_pairs], index=plot_df.index)
    bad_lots = {lot for lot, _ in bad_pairs}
    others = plot_df[~is_bad_row]
    bad = plot_df[is_bad_row]

    fig = go.Figure()

    def add_group(grp: pd.DataFrame, color: str, name: str, is_bad: bool = False) -> None:
        if grp.empty:
            return
        marker = (
            dict(color="red", size=11, line=dict(width=1, color="black"))
            if is_bad
            else dict(color=color, size=7)
        )
        fig.add_trace(
            go.Scatter(
                x=grp["tkout_time"], y=grp[item_id],
                mode="markers", marker=marker,
                name=name,
                # the held wafer is added last so it draws on top, but ranks
                # first in the legend: with many categories plotly clips the
                # legend, and this entry must never be the one cut off
                legendrank=1 if is_bad else 1000 + len(fig.data),
                customdata=grp[HOVER_COLS].values,
                hovertemplate=HOVER_TEMPLATE,
            )
        )

    if legend_field is None:
        # other wafers from the held lot(s) are the most useful comparison,
        # so they get a darker gray than the rest of the population
        same_lot_mask = others["root_lot_id"].map(norm_lot).isin(bad_lots)
        add_group(others[~same_lot_mask], "lightgray", "other")
        add_group(others[same_lot_mask], "dimgray", ",".join(sorted(bad_lots)))
    else:
        # dropna=False: rows whose legend field is blank would otherwise be
        # dropped from every group and silently vanish from the chart
        for i, (cat_val, grp) in enumerate(others.groupby(legend_field, dropna=False)):
            label = "(없음)" if pd.isna(cat_val) else str(cat_val)
            add_group(grp, CATEGORY_COLORS[i % len(CATEGORY_COLORS)], label)

    if legend_field is None:
        add_group(bad, None, bad_label, is_bad=True)
    else:
        for cat_val, grp in bad.groupby(legend_field, dropna=False):
            label = "(없음)" if pd.isna(cat_val) else str(cat_val)
            add_group(grp, None, f"{bad_label}_{label}", is_bad=True)

    # skip any limit that didn't parse to a number rather than passing None
    # through to plotly, which errors on a missing y just as it does on a str
    if ucl is not None:
        fig.add_hline(y=ucl, line=dict(color="blue", dash="dash"), annotation_text="UCL", annotation_position="top left")
    if lcl is not None:
        fig.add_hline(y=lcl, line=dict(color="blue", dash="dash"), annotation_text="LCL", annotation_position="bottom left")
    if usl is not None:
        fig.add_hline(y=usl, line=dict(color="red", dash="dash"), annotation_text="USL", annotation_position="top left")
    if lsl is not None:
        fig.add_hline(y=lsl, line=dict(color="red", dash="dash"), annotation_text="LSL", annotation_position="bottom left")

    fig.update_layout(
        xaxis_title="tkout_time",
        yaxis_title=item_id,
        legend_title=legend_field or "Legend",
        # compact legend so a high-cardinality field still fits without
        # plotly clipping entries off the bottom
        legend=dict(font=dict(size=10), itemsizing="constant", tracegroupgap=0),
        height=chart_height,
        margin=dict(t=30, b=30),
    )
    return fig


st.markdown(
    f"# Hold 현황 "
    f"<span style='font-size:0.42em; font-weight:400; color:#888;'>"
    f"(Latest Data : {DATA_LOADED_AT})</span>",
    unsafe_allow_html=True,
)

PANEL_HEIGHT = 650
TREND_HEIGHT = round(PANEL_HEIGHT * 2 / 3)
COMMENT_HEIGHT = PANEL_HEIGHT - TREND_HEIGHT
# the left column splits the same total between the grouped list and the
# breakdown of whichever lot is selected, so both columns still end level
LIST_HEIGHT = 400
DETAIL_HEIGHT = PANEL_HEIGHT - LIST_HEIGHT
DETAIL_COLS = ["wafer_id", "item_id", "rw_cnt", "hold_inform"]

left, right = st.columns([2, 3])

with left:
    title_col, switch_col = st.columns([2, 2])
    with title_col:
        st.subheader("DC OCAP List")
    with switch_col:
        # required=True: without it, clicking the active product deselects it
        # and the list silently falls back to ULY with no product highlighted
        selected_product = st.segmented_control(
            "제품", list(PRODUCT_DC.keys()), default="ULY", required=True,
            label_visibility="collapsed", key="product_switch",
        )
    selected_product = selected_product or "ULY"

    dc_df = PRODUCT_DC[selected_product]
    if dc_df is None or dc_df.empty or "hold_time" not in dc_df.columns:
        dc_df = pd.DataFrame(columns=DC_REQUIRED)
        st.caption(f"{selected_product}: 현재 hold 건이 없습니다.")
    grouped = group_holds(dc_df)

    event = st.dataframe(
        grouped[GROUP_COLS],
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=LIST_HEIGHT,
        key=f"dc_table_{selected_product}",
    )
    selected_rows = event.selection.rows if event and event.selection else []

    # the grouped row only says "item1외 2건", so the selected lot is broken
    # back out here: which wafer was held on which item, and why
    sel_lot_id = grouped.iloc[selected_rows[0]]["lot_id"] if selected_rows else None
    with st.container(height=DETAIL_HEIGHT, border=True):
        if sel_lot_id is None:
            st.caption("행을 클릭하면 해당 lot 의 wafer / item 내역이 표시됩니다.")
        else:
            detail = dc_df[dc_df["lot_id"] == sel_lot_id].copy()
            detail["_w"] = detail["wafer_id"].map(norm_wafer).map(
                lambda v: (0, v) if isinstance(v, int) else (1, str(v))
            )
            detail = detail.sort_values(["item_id", "_w"])
            st.caption(f"{sel_lot_id} · {len(detail)}건")
            st.dataframe(
                detail[DETAIL_COLS],
                width="stretch",
                hide_index=True,
                height=DETAIL_HEIGHT - 75,
            )

with right:
    st.subheader("Item Trend")
    product = selected_product

    lot_id = sel_lot_id
    lot_rows = dc_df[dc_df["lot_id"] == lot_id] if lot_id is not None else None

    # one chart per measurement item, stepped through with the arrows
    items = list(dict.fromkeys(lot_rows["item_id"])) if lot_rows is not None else []
    nav_key = f"item_idx_{selected_product}_{lot_id}"
    item_idx = min(st.session_state.get(nav_key, 0), max(len(items) - 1, 0))
    item_id = items[item_idx] if items else None

    tdf, item_col = find_trend_df(product, item_id) if item_id is not None else (None, None)

    with st.container(height=TREND_HEIGHT, border=True):
        if lot_id is None:
            st.info("왼쪽에서 hold 행을 클릭하면 trend 차트가 표시됩니다.")
        else:
            nav_prev, nav_label, nav_next, nav_gap, legend_col = st.columns([0.6, 2.2, 0.6, 3.5, 2])
            with nav_prev:
                if st.button("◀", key=f"prev_{nav_key}", disabled=item_idx == 0, width="stretch"):
                    st.session_state[nav_key] = item_idx - 1
                    st.rerun()
            with nav_label:
                st.markdown(
                    f"<div style='text-align:center; padding-top:0.35rem;'>"
                    f"<b>{item_id}</b> <span style='color:#888;'>({item_idx + 1}/{len(items)})</span></div>",
                    unsafe_allow_html=True,
                )
            with nav_next:
                if st.button("▶", key=f"next_{nav_key}", disabled=item_idx >= len(items) - 1, width="stretch"):
                    st.session_state[nav_key] = item_idx + 1
                    st.rerun()
            with legend_col:
                legend_label = st.selectbox(
                    "Legend", list(LEGEND_FIELD_OPTIONS.keys()), index=0, label_visibility="collapsed"
                )
            legend_field = LEGEND_FIELD_OPTIONS[legend_label]

            if tdf is None:
                st.warning(f"{item_id} 에 매칭되는 trend 데이터를 찾지 못했습니다.")
            else:
                # only the wafers held for THIS item are the excursion
                item_rows = lot_rows[lot_rows["item_id"] == item_id]
                bad_pairs = set(zip(item_rows["root_lot_id"].map(norm_lot),
                                    item_rows["wafer_id"].map(norm_wafer)))
                wafer_list = ",".join(str(w) for w in sort_wafers(item_rows["wafer_id"]))
                limits = item_rows.iloc[0]

                if tdf[item_col].notna().sum() == 0:
                    st.warning(f"{item_id} 은(는) 이 제품 trend 에 측정값이 없습니다.")

                fig = build_scatter(
                    tdf, item_col, bad_pairs, f"{lot_id} #{wafer_list}", legend_field,
                    limits["ucl"], limits["lcl"], limits["usl"], limits["lsl"],
                    chart_height=TREND_HEIGHT - 150,
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    f"제품: {product} · lot_id: {lot_id} · "
                    f"wafer_id: {wafer_list} · item: {item_id}"
                )

    with st.container(height=COMMENT_HEIGHT, border=True):
        if lot_id is None:
            st.caption("Comment")
        else:
            # the disposition is recorded in the company system and merged
            # into dc, so it is shown read-only rather than edited here
            item_rows = lot_rows[lot_rows["item_id"] == item_id] if item_id else lot_rows
            st.text_area(
                "Comment",
                value="\n".join(dict.fromkeys(str(c) for c in item_rows["comment"])),
                height=COMMENT_HEIGHT - 130,
                disabled=True,
                key=f"comment_view_{nav_key}_{item_idx}",
            )
            st.markdown(
                f"<div style='font-size:0.9em; line-height:1.7;'>"
                f"<b>owner</b> &nbsp;{' / '.join(dict.fromkeys(str(o) for o in item_rows['owner']))}<br>"
                f"<b>code</b> &nbsp;&nbsp;{' / '.join(dict.fromkeys(str(c) for c in item_rows['code']))}"
                f"</div>",
                unsafe_allow_html=True,
            )

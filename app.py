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
    events = []  # (lot_id, src_lot, hold_time) already emitted, for reworks
    # holds arrive as lots, not as single wafers: one lot_id covers several
    # wafers and often several measurement items, which is what the
    # dashboard groups on
    while len(rows) < n_rows:
        # a rework re-holds a lot that was already held: same lot_id, a
        # later hold_time. That is what produces rw_cnt >= 1 below, and
        # it has to exist here or the dashboard's rework handling is never
        # exercised by the mock.
        if events and rng.random() < 0.25:
            lot_id, src_lot, prev_time = events[rng.integers(0, len(events))]
            hold_time = prev_time + timedelta(days=int(rng.integers(2, 10)))
            lot_rows = base_rows[base_rows["root_lot_id"] == src_lot]
        else:
            src_lot = base_rows.iloc[rng.integers(0, len(base_rows))]["root_lot_id"]
            lot_rows = base_rows[base_rows["root_lot_id"] == src_lot]
            lot_id = f"{src_lot}.{rng.integers(1, 9)}"
            hold_time = _random_datetime(rng, datetime(2026, 7, 1), datetime(2026, 8, 14, 23, 59, 59), 1).iloc[0]
        events.append((lot_id, src_lot, hold_time))

        n_waf = min(int(rng.integers(1, 6)), len(lot_rows))
        wafers = rng.choice(lot_rows["wafer_id"].unique(), size=min(n_waf, lot_rows["wafer_id"].nunique()), replace=False)
        items = rng.choice(item_cols, size=int(rng.integers(1, 4)), replace=False)

        # not every hold has been triaged yet in the company system - leave
        # a chunk of lots with no owner/code/comment so they still show up
        # as an open "hold" rather than already-dispositioned "이력"
        if rng.random() < 0.45:
            owner = code = comment = None
        else:
            owner = f"{rng.choice(OWNER_DEPTS)} {rng.choice(OWNER_NAMES)}"
            code = rng.choice(CODES, p=[0.55, 0.3, 0.15])
            comment = rng.choice(COMMENTS_BY_CODE[code])

        # status comes from a separate table that refreshes on time, unlike
        # owner/comment which only land the next morning. So an untriaged
        # lot can already have been flowed -- those must drop out of "hold"
        # straight away rather than sitting there for a day.
        if owner is None:
            status = rng.choice(["Hold", "Active", "Run"], p=[0.7, 0.2, 0.1])
        else:
            status = rng.choice(["Active", "Run", "Hold"], p=[0.6, 0.3, 0.1])

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
                rows.append(
                    {
                        "lot_id": lot_id,
                        "root_lot_id": src_lot,
                        "wafer_id": wafer_id,
                        # rw_cnt is filled in below, once every event exists
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
                        # from a separate, more frequently refreshed table
                        "status": status,
                    }
                )

    df = pd.DataFrame(rows)
    # rw_cnt the way the real pipeline derives it: within one lot_id, rank
    # the distinct hold_time values, so every row of the same hold event
    # shares a number and a re-hold after rework gets the next one. It is a
    # property of the hold event, not of an individual wafer measurement --
    # the dashboard groups the list on (lot_id, rw_cnt) and would split one
    # event into several rows otherwise.
    df["rw_cnt"] = (
        df.groupby("lot_id", observed=True)["hold_time"].rank(method="dense").astype(int) - 1
    )
    df = df[
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
            "status",
        ]
    ]
    # the random draws above can leave nothing that qualifies as "hold" for
    # an unlucky seed - force the most recent lot to qualify so the default
    # filter always has something to show. Needs status too, since a lot
    # that has been flowed is no longer a hold however blank its comment is.
    if not df.empty and not (
        df["owner"].isna() & df["status"].astype(str).str.strip().eq("Hold")
    ).any():
        last_idx = df["hold_time"].idxmax()
        last_lot, last_rw_cnt = df.loc[last_idx, ["lot_id", "rw_cnt"]]
        # scoped to this one rw_cnt event, not the whole lot_id -- the lot may
        # have an earlier, already-dispositioned event that must stay intact
        mask = (df["lot_id"] == last_lot) & (df["rw_cnt"] == last_rw_cnt)
        df.loc[mask, ["owner", "code", "comment"]] = None
        df.loc[mask, "status"] = "Hold"
    return df


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
import base64
import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import streamlit as st

# KST is pinned at UTC+9 rather than read from the host clock, so the
# header timestamp stays correct wherever the app is deployed.
KST = timezone(timedelta(hours=9))

# st.session_state is one flat namespace shared by every page the portal
# renders, so every key this page touches is prefixed -- otherwise a
# generic name like "status_filter" could collide with another page's key
KEY_PREFIX = "dc_ocap_"

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
# The grouped hold list shown on the left, one row per (lot_id, rw_cnt).
# status is deliberately not here: it decides whether a row belongs in
# hold or 이력 (see filter_by_status), but a flowed lot turns into ship a
# few days later, so as a displayed value it just goes stale.
GROUP_COLS = ["rw_cnt", "hold_time", "lot_id", "wafer_id", "item", "hold_inform", "code", "owner"]

# hold 를 실제로 푸는 사내 사이트. go/dcocap 은 사내 단축주소라서 반드시
# 스킴을 붙여야 한다 -- href="go/dcocap" 은 현재 페이지 기준 상대경로로
# 해석되어 포털 안쪽 주소로 새고, 주소창에 칠 때처럼 호스트로 풀리지 않는다.
DC_HOLD_URL = "https://go/dcocap"


def sort_wafers(values) -> list:
    """Wafer numbers in numeric order, with any non-numeric ones last."""
    normed = {norm_wafer(v) for v in values}
    nums = sorted(v for v in normed if isinstance(v, int))
    rest = sorted(str(v) for v in normed if not isinstance(v, int))
    return nums + rest


def summarize(values) -> str:
    """'item1' for one distinct value, 'item1외 2건' for several.

    Blank/NaN entries (e.g. an undispositioned lot's code/owner) are
    dropped rather than shown as the literal string "nan".
    """
    seen = list(dict.fromkeys(str(v).strip() for v in values if pd.notna(v) and str(v).strip()))
    if not seen:
        return ""
    return seen[0] if len(seen) == 1 else f"{seen[0]}외 {len(seen) - 1}건"


def format_disposition(rows: pd.DataFrame) -> tuple[str, str, str] | None:
    """Distinct (comment, owner, code) text from dc rows, or None.

    None means "there is nothing real to show": no row matched at all, or
    every row's comment and owner are both blank/NaN. The company system
    merge can leave either blank on a genuine record, so a comment with no
    owner (or vice versa) still counts as history and is shown as-is.
    """
    if rows.empty:
        return None
    comments = [str(c).strip() for c in rows["comment"] if pd.notna(c) and str(c).strip()]
    owners = [str(o).strip() for o in rows["owner"] if pd.notna(o) and str(o).strip()]
    if not comments and not owners:
        return None
    codes = [str(c).strip() for c in rows["code"] if pd.notna(c) and str(c).strip()]
    return (
        "\n".join(dict.fromkeys(comments)) if comments else "-",
        " / ".join(dict.fromkeys(owners)) if owners else "-",
        " / ".join(dict.fromkeys(codes)) if codes else "-",
    )


def filter_by_status(dc_df: pd.DataFrame, view: str) -> pd.DataFrame:
    """전체: 그대로 반환. hold: 아직 조치가 안 된 행만. 이력: 나머지 전부.

    "조치가 안 됨" 은 두 가지를 모두 만족해야 합니다:
      1) code 와 owner 가 둘 다 비어있음 (아직 코멘트가 안 달림)
      2) status 가 정확히 'Hold' 임 (H 만 대문자, 설비상 아직 잡혀 있음)

    2번이 필요한 이유: dc 는 30분마다 갱신되는데 owner/comment 는 다음날
    아침에야 적재되기 때문에, 누군가 코멘트를 달고 flow 시켜도 하루 동안
    hold 에 남아 있게 됩니다. status 는 제때 갱신되므로 이미 flow 된 건
    (Active / Run / ship 등) 을 그 자리에서 이력으로 넘길 수 있습니다.

    'Hold' 가 아닌 것은 빈 값도 포함해서 전부 이력입니다. 즉 status 가
    아직 안 붙은 신규 hold 는 기본 화면에 안 보입니다. status 결측이
    쌓이면 hold 가 통째로 비어 보이므로, diagnose.py 4번이 결측 건수를
    따로 세어 경고합니다.

    status 컬럼이 없는 dc 도 그대로 동작하도록, 없으면 1번만 봅니다.
    """
    if dc_df.empty or view == "전체":
        return dc_df
    code_blank = dc_df["code"].isna() | (dc_df["code"].astype(str).str.strip() == "")
    owner_blank = dc_df["owner"].isna() | (dc_df["owner"].astype(str).str.strip() == "")
    undispositioned = code_blank & owner_blank
    if "status" in dc_df.columns:
        # astype(str) 이 NaN 을 "nan" 으로 바꾸므로 결측도 자연히 탈락한다
        undispositioned &= dc_df["status"].astype(str).str.strip().eq("Hold")
    return dc_df[undispositioned] if view == "hold" else dc_df[~undispositioned]


def count_new_holds(dc_df: pd.DataFrame) -> int:
    """How many rows the list shows under the "hold" filter.

    Deliberately runs the list's own filter and grouping rather than
    re-deriving anything here, so the header count can never drift from
    the rows underneath it -- including after grouping changed from
    lot_id alone to (lot_id, rw_cnt).
    """
    if dc_df.empty or "lot_id" not in dc_df.columns:
        return 0
    return len(group_holds(filter_by_status(dc_df, "hold")))


def group_holds(dc_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the hold list to one row per (lot_id, rw_cnt).

    A single lot is held as one event covering several wafers and often
    several measurement items, so showing one row per (wafer, item) buries
    the engineer in near-duplicate rows. Wafers are listed out ("2,3,4")
    since the count is small and it says which wafers are affected; items
    are summarized ("item1외 2건") because the chart pages through them
    one at a time anyway.

    rw_cnt is part of the key rather than summarized away: a lot held
    again after a rework is a separate event with its own hold_time and
    its own disposition, so collapsing both into one row would hide the
    later one entirely.
    """
    if dc_df.empty or "lot_id" not in dc_df.columns:
        return pd.DataFrame(columns=GROUP_COLS)

    # tolerate dc without rw_cnt rather than dying on the groupby
    keys = ["lot_id", "rw_cnt"] if "rw_cnt" in dc_df.columns else ["lot_id"]

    rows = []
    for key, grp in dc_df.groupby(keys, dropna=False, sort=False):
        # pandas hands back a 1-tuple when grouping on a one-element list
        # (and a bare scalar on older versions), so normalize before indexing
        key = key if isinstance(key, tuple) else (key,)
        lot_id = key[0]
        rw_cnt = key[1] if len(key) > 1 else ""
        rows.append({
            "rw_cnt": rw_cnt,
            "hold_time": grp["hold_time"].max(),
            "lot_id": lot_id,
            "wafer_id": ",".join(str(w) for w in sort_wafers(grp["wafer_id"])),
            "item": summarize(grp["item_id"]),
            "hold_inform": summarize(grp["hold_inform"]),
            "code": summarize(grp["code"]),
            "owner": summarize(grp["owner"]),
        })
    # newest first as before; rw_cnt descending breaks ties so a rework
    # (rw_cnt 1) sits above the original (rw_cnt 0) even when the company
    # system stamped both with the same hold_time
    return (
        pd.DataFrame(rows)
        .sort_values(["hold_time", "rw_cnt"], ascending=[False, False])
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


def norm_rw_cnt(value) -> str:
    """Canonical rw_cnt for matching a list row back to its own dc rows.

    The list groups on (lot_id, rw_cnt), and everything on the right --
    trend, control limits, comment -- is then looked up with that pair. A
    plain `==` breaks that lookup twice over: a missing rw_cnt is NaN, and
    NaN never equals itself, so the row sits in the list with an empty
    chart beside it; and 0 vs 0.0 miss each other whenever one frame came
    back float and the other int. Both collapse to one text key here, with
    blank/NaN treated as its own bucket rather than dropped.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "":
        return ""
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


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


def check_data(product_dc: dict, trend_frames: dict) -> tuple[list[str], list[str]]:
    """Return (fatal, warnings) about what pull_data() handed back.

    Runs on the real data the first time it is plugged in, so a schema
    mismatch reads as a plain list of what to fix instead of a KeyError.
    Called from inside the cached load_data(), because scanning full
    trend tables on every rerun would cost more than the checks are
    worth (~0.8s per click on 200k rows).

    The split matters: only things that actually stop the dashboard from
    working belong in `fatal`, because the caller refuses to render on
    those. A wiring mistake (wrong column, wrong dtype, nothing matching
    at all) is fatal. Real-world data mess -- a handful of wafers that
    never made it into trend, an item with no trend column -- is a
    warning, because the page handles it: the chart simply has no red dot
    or shows its own "not found" box, and blocking the whole dashboard
    over one row out of hundreds is worse than the missing dot.
    """
    fatal, warnings = [], []
    problems = fatal  # 기존 검사들이 쓰던 이름 유지 (치명적 목록)
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
                # build_scatter 가 to_float 로 한 번 더 거르므로 그리다 죽지는
                # 않고, 해당 행의 관리선만 안 그려진다 -> 경고로 충분
                warnings.append(
                    f"dc_{product.lower()}: {limit_col} 의 {bad_count}개 값이 숫자로 "
                    f"변환되지 않습니다 (dtype={dc_df[limit_col].dtype}). "
                    f"해당 hold 는 관리선이 안 그려집니다. float 로 변환하세요."
                )

        # dc.item_id must name a real column in that product's trend
        # (case-insensitively -- resolve_item_col handles the difference)
        if "item_id" in dc_df.columns and not dc_df.empty:
            all_items = set(dc_df["item_id"])
            unknown = sorted(
                str(i) for i in all_items if resolve_item_col(trend_df, i) is None
            )
            if unknown and len(unknown) == len(all_items):
                # 하나도 안 맞으면 컬럼명 규칙 자체가 어긋난 것 (배선 실수)
                problems.append(
                    f"dc_{product.lower()}: item_id 가 {product.lower()}_trend 의 "
                    f"컬럼과 하나도 매칭되지 않습니다. dc 예시 {unknown[:5]}, "
                    f"trend 컬럼 예시 {[str(c) for c in trend_df.columns[:5]]}"
                )
            elif unknown:
                # 일부만 없는 것은 흔한 일이고, 그 item 을 고르면 화면이
                # "매칭되는 trend 데이터를 찾지 못했습니다" 를 직접 띄운다
                warnings.append(
                    f"dc_{product.lower()}: item_id {unknown[:5]} 이(가) "
                    f"{product.lower()}_trend 의 컬럼에 없습니다 "
                    f"(총 {len(unknown)}/{len(all_items)}종, 해당 item 은 차트가 안 뜸)."
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
                # 일부만 없는 것: 그 wafer 만 빨간 점이 안 찍히고 나머지는
                # 정상이다. bad_pairs 는 단순 집합 조회라 죽지도 않는다.
                warnings.append(
                    f"dc_{product.lower()}: {len(missing)}/{len(dc_pairs)} 건의 "
                    f"(root_lot_id, wafer_id) 가 {product.lower()}_trend 에 없습니다 "
                    f"(해당 hold 는 빨간 점이 안 찍힘). 예시 {sorted(missing)[:3]}"
                )
    return fatal, warnings


# Streamlit re-runs show_dc_ocap() on every click (the portal reruns its
# whole script top to bottom, same as any Streamlit app), so pull_data()
# is called through a cache -- otherwise every row selection would
# re-query the company system. ttl is how stale the data may get before
# the next interaction refetches it; raise or lower it to taste, and use
# the app's ⋮ menu > Clear cache to force a refresh.
@st.cache_data(ttl=600, show_spinner="데이터 불러오는 중...")
def load_data():
    frames = pull_data()
    dc_uly, dc_sol, dc_tts, uly_trend, sol_trend, tts_trend = frames
    # checked here rather than on every rerun: it scans the whole trend
    # tables, which is far too slow to repeat on each click
    problems, warnings = check_data(
        {"ULY": dc_uly, "SOL": dc_sol, "TTS": dc_tts},
        {"ULY": uly_trend, "SOL": sol_trend, "TTS": tts_trend},
    )
    # stamped inside the cache, so the header reports when the data was
    # actually fetched rather than when the page was last re-rendered
    loaded_at = datetime.now(KST).strftime("%y/%m/%d %H:%M")
    return (*frames, loaded_at, problems, warnings)


LEGEND_FIELD_OPTIONS = {
    "없음 (기본)": None,
    "probe_card_id": "probe_card_id",
    "eqp_id": "eqp_id",
    "lot_type": "lot_type",
    "rw_cnt": "rw_cnt",
}

# "_hover_time" rather than "tkout_time" itself: the x-axis needs the real
# datetime column, and a plain string reads better in the hover box than
# whatever plotly would stringify a raw Timestamp to
HOVER_COLS = ["root_lot_id", "wafer_id", "_hover_time", "probe_card_id", "eqp_id", "lot_type", "rw_cnt"]
HOVER_TEMPLATE = (
    "root_lot_id=%{customdata[0]}<br>"
    "wafer_id=%{customdata[1]}<br>"
    "tkout_time=%{customdata[2]}<br>"
    "probe_card_id=%{customdata[3]}<br>"
    "eqp_id=%{customdata[4]}<br>"
    "lot_type=%{customdata[5]}<br>"
    "rw_cnt=%{customdata[6]}<extra></extra>"
)

# red means "past the scrap limit" and blue "past the control limit", so
# both hues -- and anything close enough to be mistaken for them at marker
# size, like orange or light blue -- are kept out of this palette. 20
# entries so a high-cardinality field (many probe cards / eqp ids in the
# queried window) doesn't wrap onto a duplicate color too quickly.
LIMIT_COLORS = {"scrap": "red", "control": "blue"}
CATEGORY_COLORS = [
    "#2ca02c", "#9467bd", "#8c564b", "#bcbd22", "#17becf",
    "#e377c2", "#7f7f7f", "#1b9e77", "#a6761d", "#66a61e",
    "#5d4037", "#8e6c8a", "#93a01e", "#4d4d4d", "#c49a6c",
    "#7fbf7b", "#af8dc3", "#d9a441", "#2f6f4e", "#6b4f8a",
]


def find_trend_df(trend_frames: dict, product: str, item_id: str):
    """Return (trend dataframe, its column for item_id), or (None, None).

    The product is known from the hold list's own selection, so it is
    used directly instead of searching every trend frame for a matching
    root_lot_id -- a lot that appears under more than one product would
    otherwise chart the wrong product's data. The column is resolved
    rather than taken literally because dc and the trend tables can
    capitalise item ids differently. trend_frames is passed in rather
    than read off a module global, since that global only exists once
    show_dc_ocap() has actually loaded the data for this run.
    """
    tdf = trend_frames.get(product)
    if tdf is None or tdf.empty:
        return None, None
    item_col = resolve_item_col(tdf, item_id)
    if item_col is None:
        return None, None
    return tdf, item_col


def build_scatter(trend_df, item_id: str, bad_pairs: set, bad_label: str,
                   legend_field: str | None, ucl, lcl, usl, lsl,
                   chart_height: int, focus_pair: tuple | None = None) -> go.Figure:
    """Scatter one measurement item over time.

    `bad_pairs` is the set of normalized (root_lot_id, wafer_id) pairs held
    for this item -- a lot is held as a whole, so its wafers are
    highlighted together rather than one legend entry each. Those held
    wafers are then split by which limit they broke; `bad_label` is the
    lot id the entries are named after.

    `focus_pair`, if given, is the (root_lot_id, wafer_id) last clicked in
    the chart -- it gets a black ring on top of its existing marker(s) and
    its own legend entry, so the selection stays visible regardless of
    which color group the point itself belongs to.
    """
    ucl, lcl, usl, lsl = to_float(ucl), to_float(lcl), to_float(usl), to_float(lsl)

    # coerce before dropna: a numeric column that came back as text/object
    # (BigQuery NUMERIC/DECIMAL) would otherwise keep its non-null string
    # values and only fail once plotly tries to lay out the chart
    plot_df = trend_df.assign(**{item_id: pd.to_numeric(trend_df[item_id], errors="coerce")})
    plot_df = plot_df.dropna(subset=[item_id])
    # keep tkout_time itself as a real datetime (needed for the x-axis);
    # format a separate string column just for the hover box
    plot_df["_hover_time"] = plot_df["tkout_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # match on the pair (wafer numbers 1-25 repeat across lots), and
    # normalize both sides: dc and the trend table need not agree on how
    # a lot id is padded or whether the wafer number is text or an int
    row_pairs = list(zip(plot_df["root_lot_id"].map(norm_lot),
                         plot_df["wafer_id"].map(norm_wafer)))
    is_bad_row = pd.Series([p in bad_pairs for p in row_pairs], index=plot_df.index)
    bad_lots = {lot for lot, _ in bad_pairs}

    # a point past the scrap limit is necessarily past the control limit
    # too, so scrap wins and the two groups stay disjoint
    values = plot_df[item_id]
    past_scrap = pd.Series(False, index=plot_df.index)
    if usl is not None:
        past_scrap |= values > usl
    if lsl is not None:
        past_scrap |= values < lsl
    past_control = pd.Series(False, index=plot_df.index)
    if ucl is not None:
        past_control |= values > ucl
    if lcl is not None:
        past_control |= values < lcl
    past_control &= ~past_scrap

    bad = plot_df[is_bad_row]
    in_spec = ~(past_scrap | past_control)

    # exact-zero readings are almost always measurement glitches, not real
    # excursions -- dropped from the background context points only, so a
    # handful of them don't blow out the y-axis. Held wafers are exempt: a
    # hold is often triggered by exactly this kind of extreme value, and
    # hiding it would hide the reason it was held in the first place.
    others = plot_df[~is_bad_row & (plot_df[item_id] != 0)]

    fig = go.Figure()

    def add_group(grp: pd.DataFrame, color: str, name: str, rank: int, is_bad: bool = False) -> None:
        if grp.empty:
            return
        # held wafers keep the bigger outlined marker so they stay findable;
        # their fill says which limit the point broke
        marker = (
            dict(color=color, size=11, line=dict(width=1, color="black"))
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
                legendrank=rank,
                customdata=grp[HOVER_COLS].values,
                hovertemplate=HOVER_TEMPLATE,
            )
        )

    if legend_field is None:
        # other wafers from the held lot(s) are the most useful comparison,
        # so they get a darker gray than the rest of the population
        same_lot_mask = others["root_lot_id"].map(norm_lot).isin(bad_lots)
        add_group(others[~same_lot_mask], "lightgray", "other", rank=1100)
        add_group(others[same_lot_mask], "dimgray", ",".join(sorted(bad_lots)), rank=1050)
    else:
        # dropna=False: rows whose legend field is blank would otherwise be
        # dropped from every group and silently vanish from the chart
        for i, (cat_val, grp) in enumerate(others.groupby(legend_field, dropna=False)):
            label = "(없음)" if pd.isna(cat_val) else str(cat_val)
            add_group(grp, CATEGORY_COLORS[i % len(CATEGORY_COLORS)], label, rank=1000 + i)

    # the held wafers are split by which limit they broke -- that judgement
    # is what the engineer is here to make. Scrap outranks control in the
    # legend, and a held wafer inside both limits (held on a trend rule or
    # an equipment alarm) is drawn hollow rather than given a limit color.
    # The entries are named by lot and wafer only: the marker color already
    # says which limit, so spelling it out again just crowds the legend.
    def add_bad(subset: pd.DataFrame, color: str, rank: int) -> None:
        if subset.empty:
            return
        wafers = ",".join(str(w) for w in sort_wafers(subset["wafer_id"]))
        add_group(subset, color, f"{bad_label} #{wafers}", rank=rank, is_bad=True)

    add_bad(bad[past_scrap.loc[bad.index]], LIMIT_COLORS["scrap"], 1)
    add_bad(bad[past_control.loc[bad.index]], LIMIT_COLORS["control"], 2)
    add_bad(bad[in_spec.loc[bad.index]], "white", 3)

    # ring around the last-clicked wafer, drawn last (so it's on top) and
    # ranked first in the legend -- it doesn't replace the point's own
    # color, just marks which one is currently focused
    if focus_pair is not None:
        froot, fwafer = focus_pair
        focus_rows = plot_df[
            plot_df["root_lot_id"].map(norm_lot).eq(norm_lot(froot))
            & plot_df["wafer_id"].map(norm_wafer).eq(norm_wafer(fwafer))
        ]
        if not focus_rows.empty:
            fig.add_trace(
                go.Scatter(
                    x=focus_rows["tkout_time"], y=focus_rows[item_id],
                    mode="markers",
                    marker=dict(color="rgba(0,0,0,0)", size=18, line=dict(width=3, color="black")),
                    name=f"선택 WF: {norm_lot(froot)} #{norm_wafer(fwafer)}",
                    legendrank=0,
                    customdata=focus_rows[HOVER_COLS].values,
                    hovertemplate=HOVER_TEMPLATE,
                )
            )

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


# ====================================================================
# 🖥️ Streamlit 메인 화면 UI 구성 (DC OCAP List + Item Trend)
# ====================================================================
def show_dc_ocap():
    """Render the DC OCAP hold dashboard as a portal page.

    Left: the selected product's hold list, grouped to one row per
    lot_id, newest first, single-row selectable, with a ULY/TTS/SOL
    switch and a 전체/hold/이력 status filter.
    Right (top 2/3): scatter of one measurement item across that
    product's trend dataframe, with UCL/LCL (blue) and USL/LSL (red)
    reference lines; clicking a wafer point rings it, adds it to the
    legend, and switches the left list to that wafer's own lot.
    Right (bottom 1/3): the disposition recorded in the company system
    and merged into dc -- comment, then owner and code -- read-only.

    Assumes the caller (portal.py) has already run st.set_page_config
    with layout="wide" -- that call can only happen once per app and
    must be the first streamlit command, so it doesn't belong in here.
    """
    (dc_uly, dc_sol, dc_tts, uly_trend, sol_trend, tts_trend,
     data_loaded_at, problems, data_warnings) = load_data()

    trend_frames = {"ULY": uly_trend, "SOL": sol_trend, "TTS": tts_trend}
    product_dc = {"ULY": dc_uly, "TTS": dc_tts, "SOL": dc_sol}

    if problems:
        st.error("pull_data() 가 돌려준 데이터가 대시보드 형식과 맞지 않습니다:")
        for p in problems:
            st.write("- " + p)
        st.stop()

    # 치명적이지 않은 것들은 접어서 보여준다. 일부 wafer 가 trend 에 없는
    # 정도로 대시보드 전체를 막으면, 정작 멀쩡한 나머지 hold 를 못 본다.
    if data_warnings:
        with st.expander(f"⚠️ 데이터 참고사항 {len(data_warnings)}건 (대시보드는 정상 동작)"):
            for w in data_warnings:
                st.write("- " + w)

    # the counts sit inside the h1 so their 0.42em resolves against the same
    # heading size -- that keeps them consistent without having to hardcode
    # whatever px streamlit's h1 currently renders at. "Latest Data" used to
    # sit here too but now rides just above the trend panel, next to the
    # DC HOLD link.
    meta_style = "font-size:0.42em; font-weight:400; color:#888; line-height:1.35;"
    new_counts = ", ".join(f"{p} : {count_new_holds(product_dc[p])}건" for p in product_dc)
    st.markdown(
        f"# Hold 현황 "
        f"<span style='display:inline-block; vertical-align:bottom; {meta_style}'>"
        f"신규 hold 건수<br>{new_counts}</span>",
        unsafe_allow_html=True,
    )

    # the comment box is a disabled (read-only) text_area, which streamlit
    # renders in light gray by default; override to black and slightly larger
    # so it's actually legible. -webkit-text-fill-color is needed too since
    # some browsers ignore `color` on a disabled field and only honor this.
    st.markdown(
        """
        <style>
        div[data-testid="stTextArea"] textarea:disabled {
            color: #000 !important;
            -webkit-text-fill-color: #000 !important;
            font-size: 1.05rem !important;
        }
        /* shrink the status-filter / product-switch segmented controls so the
           list title and both of them fit on a single row */
        div[data-testid="stButtonGroup"] button[data-variant="segmented_control"] {
            padding: 0.15rem 0.55rem !important;
            min-height: unset !important;
        }
        div[data-testid="stButtonGroup"] button[data-variant="segmented_control"] p {
            font-size: 0.8rem !important;
        }
        /* DC HOLD 바로가기. 사내 단축주소로 나가는 링크라 버튼처럼 보이게
           네모 박스로 그린다 (st.link_button 은 이 자리에서 폭/여백이
           제멋대로라 마크업으로 직접 그림) */
        a.dc-hold-link {
            /* line-height 를 못 박아 둔다. 기본값이면 브라우저마다 버튼
               높이가 달라져 왼쪽 컬럼과 헤더 높이를 맞춰둔 게 어긋난다 */
            display: inline-block; padding: 3px 14px; line-height: 1.35;
            border: 1px solid #d0d3d9; border-radius: 8px;
            background: #fff; color: #d33 !important;
            font-size: 0.85rem; font-weight: 700; text-decoration: none !important;
        }
        a.dc-hold-link:hover { background: #fff1f0; border-color: #d33; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    PANEL_HEIGHT = 650
    # the left column splits the total between the grouped list and the
    # breakdown of whichever lot is selected
    LIST_HEIGHT = 400
    DETAIL_HEIGHT = PANEL_HEIGHT - LIST_HEIGHT
    # trend matches the list box beside it, so comment matches the breakdown --
    # both columns still add up to PANEL_HEIGHT and end level. This works
    # because the right header is two rows now (link, then title + timestamp),
    # same as the left's (title + filter, then product switch)
    TREND_HEIGHT = LIST_HEIGHT
    COMMENT_HEIGHT = PANEL_HEIGHT - TREND_HEIGHT
    DETAIL_COLS = ["rw_cnt", "wafer_id", "item_id", "hold_inform"]
    # tracks the last wafer point clicked in the trend chart, independent of
    # any one widget's key, so it survives the list/nav-index switch a click
    # can trigger (which would otherwise remount the chart and lose its own
    # state)
    focus_key = f"{KEY_PREFIX}focused_wafer"

    left, right = st.columns([2, 3])

    with left:
        title_col, status_col, switch_col = st.columns([1.3, 1.75, 1.6])
        with title_col:
            st.markdown(
                "<div style='font-size:1.15rem; font-weight:700; padding-top:0.3rem;'>"
                "DC OCAP List</div>",
                unsafe_allow_html=True,
            )
        with status_col:
            # required=True: same reasoning as the product switch below - without
            # it, clicking the active option deselects it instead of staying put
            status_filter = st.segmented_control(
                "상태", ["전체", "hold", "이력"], default="hold", required=True,
                label_visibility="collapsed", key=f"{KEY_PREFIX}status_filter",
            )
        with switch_col:
            # required=True: without it, clicking the active product deselects it
            # and the list silently falls back to ULY with no product highlighted
            # width="stretch": fills the column so it lands flush with the right
            # edge of the list table below, matching the requested layout
            selected_product = st.segmented_control(
                "제품", list(product_dc.keys()), default="ULY", required=True,
                label_visibility="collapsed", key=f"{KEY_PREFIX}product_switch", width="stretch",
            )
        selected_product = selected_product or "ULY"
        status_filter = status_filter or "hold"

        full_dc_df = product_dc[selected_product]
        if full_dc_df is None or full_dc_df.empty or "hold_time" not in full_dc_df.columns:
            full_dc_df = pd.DataFrame(columns=DC_REQUIRED)
            st.caption(f"{selected_product}: 현재 hold 건이 없습니다.")
        # a clicked chart point is resolved against the full (unfiltered) set,
        # since its history shouldn't disappear just because the current status
        # filter happens to hide the lot it belongs to
        dc_df = filter_by_status(full_dc_df, status_filter)
        grouped = group_holds(dc_df)

        # a chart click on the previous run may have asked to switch the list's
        # own selection to a different lot; that has to happen here, before the
        # dataframe widget below is instantiated, since a widget's session_state
        # key can no longer be written once the widget itself has been created
        table_key = f"{KEY_PREFIX}dc_table_{selected_product}"
        pending_switch = st.session_state.pop(f"{KEY_PREFIX}pending_lot_switch", None)
        if pending_switch and pending_switch.get("product") == selected_product:
            match_idx = grouped.index[
                (grouped["lot_id"] == pending_switch["lot_id"])
                & grouped["rw_cnt"].map(norm_rw_cnt).eq(norm_rw_cnt(pending_switch["rw_cnt"]))
            ]
            if len(match_idx):
                st.session_state[table_key] = {
                    "selection": {"rows": [int(match_idx[0])], "columns": []}
                }

        event = st.dataframe(
            grouped[GROUP_COLS],
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=LIST_HEIGHT,
            key=table_key,
        )
        selected_rows = event.selection.rows if event and event.selection else []

        # the grouped row only says "item1외 2건", so the selected lot is broken
        # back out here: which wafer was held on which item, and why. rw_cnt
        # comes along too -- the same lot_id can have several hold events, and
        # the trend/limits below must stick to the one that was clicked, not
        # every event the lot has ever had
        sel_lot_id = grouped.iloc[selected_rows[0]]["lot_id"] if selected_rows else None
        sel_rw_cnt = grouped.iloc[selected_rows[0]]["rw_cnt"] if selected_rows else None
        with st.container(height=DETAIL_HEIGHT, border=True):
            if sel_lot_id is None:
                st.caption("행을 클릭하면 해당 lot 의 wafer / item 내역이 표시됩니다.")
            else:
                # every rw_cnt of the lot, not just the one the clicked row
                # stands for -- seeing the original next to its rework is the
                # point of the breakdown. Read from the unfiltered frame, or
                # the hold view would hide the already-dispositioned pass and
                # leave only the row that was clicked.
                detail = full_dc_df[full_dc_df["lot_id"] == sel_lot_id].copy()
                # astype(object) first: mapping a category column twice makes
                # pandas try to rebuild it as a category, and since this map's
                # result is tuples, pandas mistakes them for a MultiIndex and
                # raises inside `.hasnans` instead of just building the column
                detail["_w"] = detail["wafer_id"].astype(object).map(norm_wafer).map(
                    lambda v: (0, v) if isinstance(v, int) else (1, str(v))
                )
                # rework on top, then wafers ascending, items grouped per wafer
                sort_cols = (["rw_cnt"] if "rw_cnt" in detail.columns else []) + ["_w", "item_id"]
                ascending = ([False] if "rw_cnt" in detail.columns else []) + [True, True]
                detail = detail.sort_values(sort_cols, ascending=ascending)
                st.caption(f"{sel_lot_id} · {len(detail)}건")
                st.dataframe(
                    detail[DETAIL_COLS],
                    width="stretch",
                    hide_index=True,
                    height=DETAIL_HEIGHT - 75,
                )

    with right:
        # 제목과 링크/데이터 시각을 한 줄에 둔다. st.subheader 를 따로 쓰면
        # 제목과 이 블록이 두 줄로 쌓여 패널이 그만큼 아래로 밀리므로, 하나의
        # flex 로 직접 그린다. align-items:flex-end 라 오른쪽 블록의 아랫줄이
        # 제목 밑선과 나란히 떨어진다.
        # href 는 반드시 절대주소(https://go/...)여야 한다 -- "go/dcocap" 만
        # 쓰면 현재 페이지 기준 상대경로로 붙어서 포털 안쪽 주소로 새고,
        # 주소창에 칠 때처럼 호스트명으로 풀리지 않는다.
        # 오른쪽 블록의 text-align:right 가 trend 패널 오른쪽 끝선을 맞춘다.
        # 한 블록으로 그린다. st.subheader 를 쓰면 그 자체가 별도 블록이라
        # 링크/시각과 같은 줄에 못 오고, 블록마다 streamlit 이 여백을 넣어
        # 왼쪽 컬럼보다 헤더가 훨씬 두꺼워진다. 폰트 크기는 정적 리포트의
        # h2(1.4rem)와 맞춰 두 화면이 같아 보이게 한다.
        st.markdown(
            f"<div style='margin-bottom:0;'>"
            f"<div style='text-align:right;'>"
            f"<a class='dc-hold-link' href='{DC_HOLD_URL}' target='_blank' rel='noopener'>"
            f"DC HOLD LINK</a></div>"
            f"<div style='display:flex; align-items:flex-end; "
            f"justify-content:space-between; gap:1rem;'>"
            f"<div style='font-size:1.4rem; font-weight:700; line-height:1;'>Item Trend</div>"
            f"<div style='font-size:0.8rem; color:#888;'>"
            f"(Latest Data : {data_loaded_at})</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        product = selected_product

        lot_id = sel_lot_id
        rw_cnt = sel_rw_cnt
        # scoped to the clicked event's rw_cnt too, not just its lot_id -- a
        # rework shares the lot_id with its original hold, and without this
        # the item list, control limits and paging state below would mix the
        # two events together
        lot_rows = (
            dc_df[
                (dc_df["lot_id"] == lot_id)
                & dc_df["rw_cnt"].map(norm_rw_cnt).eq(norm_rw_cnt(rw_cnt))
            ]
            if lot_id is not None else None
        )

        # one chart per measurement item, stepped through with the arrows
        items = list(dict.fromkeys(lot_rows["item_id"])) if lot_rows is not None else []
        nav_key = f"{KEY_PREFIX}item_idx_{selected_product}_{lot_id}_{norm_rw_cnt(rw_cnt)}"
        item_idx = min(st.session_state.get(nav_key, 0), max(len(items) - 1, 0))
        item_id = items[item_idx] if items else None

        tdf, item_col = find_trend_df(trend_frames, product, item_id) if item_id is not None else (None, None)

        # the focused wafer only applies while looking at the same product/item
        # it was clicked on -- paging away (or switching product) falls back to
        # the lot-level view, same as if nothing had been clicked
        _focus = st.session_state.get(focus_key)
        chart_focus_pair = None
        if _focus and _focus["product"] == product and _focus["item_id"] == item_id:
            chart_focus_pair = (_focus["root_lot_id"], _focus["wafer_id"])

        with st.container(height=TREND_HEIGHT, border=True):
            if lot_id is None:
                st.info("왼쪽에서 hold 행을 클릭하면 trend 차트가 표시됩니다.")
            else:
                nav_prev, nav_label, nav_next, nav_gap, legend_col, reset_col = st.columns(
                    [0.6, 2.2, 0.6, 1.8, 2, 1.6]
                )
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
                        "Legend", list(LEGEND_FIELD_OPTIONS.keys()), index=0,
                        label_visibility="collapsed", key=f"{KEY_PREFIX}legend_select_{nav_key}",
                    )
                legend_field = LEGEND_FIELD_OPTIONS[legend_label]
                rev_key = f"{KEY_PREFIX}chart_rev_{nav_key}_{item_idx}"
                with reset_col:
                    # bumps this chart's uirevision so plotly drops any zoom/pan
                    # back to the layout default, without remounting the widget
                    # (which would also throw away the clicked-point selection).
                    # This can't live inside plotly's own modebar next to zoom/pan:
                    # a modebar button's click handler has to be a JS function, and
                    # st.plotly_chart's `config` only carries JSON to the frontend,
                    # so there's no way to wire it to this rerun from here.
                    if st.button("차트 초기화", key=f"reset_{rev_key}", help="확대/이동을 초기 상태로", width="stretch"):
                        st.session_state[rev_key] = st.session_state.get(rev_key, 0) + 1
                        st.rerun()

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
                        tdf, item_col, bad_pairs, str(lot_id), legend_field,
                        limits["ucl"], limits["lcl"], limits["usl"], limits["lsl"],
                        chart_height=TREND_HEIGHT - 150, focus_pair=chart_focus_pair,
                    )
                    fig.update_layout(uirevision=st.session_state.get(rev_key, 0))

                    chart_event = st.plotly_chart(
                        fig, width="stretch",
                        on_select="rerun", selection_mode="points",
                        key=f"{KEY_PREFIX}chart_{nav_key}_{item_idx}",
                    )
                    points = chart_event.selection.points if chart_event and chart_event.selection else []
                    if points and points[0].get("customdata"):
                        cd = points[0]["customdata"]
                        new_root, new_wafer = cd[0], cd[1]  # HOVER_COLS: root_lot_id, wafer_id, ...
                        new_focus = {
                            "product": product, "item_id": item_id,
                            "root_lot_id": new_root, "wafer_id": new_wafer,
                        }
                        if new_focus != _focus:
                            st.session_state[focus_key] = new_focus
                            # if the clicked wafer belongs to a different lot than
                            # the one currently checked in the list, switch the
                            # list's own selection to match -- looked up against
                            # the unfiltered set so a status-filtered-out lot's
                            # history still resolves, even though there is then
                            # no row left to check in the (filtered) list
                            item_key = str(item_id).strip().lower()
                            click_rows = full_dc_df[
                                full_dc_df["root_lot_id"].map(norm_lot).eq(norm_lot(new_root))
                                & full_dc_df["wafer_id"].map(norm_wafer).eq(norm_wafer(new_wafer))
                                & full_dc_df["item_id"].astype(str).str.strip().str.lower().eq(item_key)
                            ]
                            # the same wafer/item can be held twice under the
                            # same lot (an original pass and its rework) -- if
                            # the click matches more than one event, stay on
                            # the one already in view rather than jumping away
                            # from under the user; otherwise prefer the most
                            # recent rework
                            if not click_rows.empty:
                                current = click_rows[
                                    (click_rows["lot_id"] == lot_id)
                                    & click_rows["rw_cnt"].map(norm_rw_cnt).eq(norm_rw_cnt(rw_cnt))
                                ]
                                pick = (
                                    current if not current.empty
                                    else click_rows.sort_values("rw_cnt", ascending=False)
                                )
                                click_lot_id = pick["lot_id"].iloc[0]
                                click_rw_cnt = pick["rw_cnt"].iloc[0]
                            else:
                                click_lot_id = None
                                click_rw_cnt = None
                            if click_lot_id is not None and (
                                (click_lot_id, norm_rw_cnt(click_rw_cnt))
                                != (lot_id, norm_rw_cnt(rw_cnt))
                            ):
                                # the list widget was already instantiated this
                                # run, so its selection can't be seeded until the
                                # next run - see pending_switch handling above
                                st.session_state[f"{KEY_PREFIX}pending_lot_switch"] = {
                                    "product": selected_product, "lot_id": click_lot_id,
                                    "rw_cnt": click_rw_cnt,
                                }
                                # keep the same item in view after the switch,
                                # if the newly-selected lot's event also holds it
                                new_lot_items = list(dict.fromkeys(
                                    dc_df[
                                        (dc_df["lot_id"] == click_lot_id)
                                        & dc_df["rw_cnt"].map(norm_rw_cnt).eq(norm_rw_cnt(click_rw_cnt))
                                    ]["item_id"]
                                ))
                                new_nav_key = (
                                    f"{KEY_PREFIX}item_idx_{selected_product}"
                                    f"_{click_lot_id}_{norm_rw_cnt(click_rw_cnt)}"
                                )
                                st.session_state[new_nav_key] = (
                                    new_lot_items.index(item_id) if item_id in new_lot_items else 0
                                )
                            st.rerun()

                    st.caption(
                        f"제품: {product} · lot_id: {lot_id} · "
                        f"wafer_id: {wafer_list} · item: {item_id}"
                    )

        with st.container(height=COMMENT_HEIGHT, border=True):
            if lot_id is None:
                st.caption("Comment")
            else:
                # the disposition is recorded in the company system and merged
                # into dc, so it is shown read-only rather than edited here.
                # Clicking a point in the chart focuses on that wafer's own
                # record instead of the currently selected lot's held wafers;
                # matching is by (root_lot_id, wafer_id, item_id) rather than
                # lot_id, since the clicked wafer may belong to a different lot
                # (or to none at all, if it was never held).
                if chart_focus_pair is not None:
                    click_lot, click_wafer = chart_focus_pair
                    item_key = str(item_id).strip().lower() if item_id else None
                    # looked up against the full (unfiltered) set: a wafer's own
                    # history shouldn't read as "없음" just because the current
                    # status filter hides the lot it belongs to
                    focus_rows = full_dc_df[
                        full_dc_df["root_lot_id"].map(norm_lot).eq(norm_lot(click_lot))
                        & full_dc_df["wafer_id"].map(norm_wafer).eq(norm_wafer(click_wafer))
                        & (full_dc_df["item_id"].astype(str).str.strip().str.lower().eq(item_key) if item_key else False)
                    ]
                    st.caption(f"선택한 wafer: {norm_lot(click_lot)} #{norm_wafer(click_wafer)}")
                else:
                    focus_rows = lot_rows[lot_rows["item_id"] == item_id] if item_id else lot_rows

                disposition = format_disposition(focus_rows)
                comment_text, owner_text, code_text = disposition or ("Comment 이력이 없습니다.", "-", "-")

                comment_key = f"{KEY_PREFIX}comment_view_{nav_key}_{item_idx}_{chart_focus_pair}"
                st.text_area(
                    "Comment",
                    value=comment_text,
                    height=COMMENT_HEIGHT - (155 if chart_focus_pair is not None else 130),
                    disabled=True,
                    key=comment_key,
                )
                st.markdown(
                    f"<div style='font-size:0.9em; line-height:1.7;'>"
                    f"<b>owner</b> &nbsp;{owner_text}<br>"
                    f"<b>code</b> &nbsp;&nbsp;{code_text}"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ======================================================================
# static-HTML export (dc_ocap.html) -- an alternative to show_dc_ocap()
# for wherever the portal serves pages from can't run a live Python
# process. Run this file directly (`python app.py`) on a schedule from
# wherever pull_data() can actually reach the company system, then
# upload the result next to the portal's other static reports (e.g. S3)
# -- this only writes the local file, since the upload step needs
# credentials this repo doesn't have.
#
# dc_ocap_template.html re-implements show_dc_ocap()'s whole interaction
# model in vanilla JS + Plotly.js by hand -- see the comment at the top
# of that file for why a straight "HTML export" of the Streamlit page
# isn't possible and this had to be a separate port instead. Both share
# pull_data()/check_data() above, so the real company-system swap only
# has to happen once.
# ======================================================================

# __file__ only exists when this runs as an actual .py script (which is
# how the scheduler will call it); it's undefined in a notebook cell, so
# fall back to the current working directory there
HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
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


def _columns(df: pd.DataFrame) -> dict:
    """DataFrame -> {column: [values]} instead of a list of one dict per
    row. Row-shaped JSON repeats every column name on every single row --
    with ~150 days of trend history and 20+ item columns that repetition
    is most of the file. Column-shaped JSON writes each name once; the
    template expands it back into per-row objects client-side (see
    expandColumnar() in dc_ocap_template.html), so nothing downstream of
    that expansion has to change."""
    return {col: [_clean(v) for v in df[col]] for col in df.columns}


def build_dc_ocap_html() -> Path:
    """Generate dc_ocap.html from the current pull_data() and return its path."""
    dc_uly, dc_sol, dc_tts, uly_trend, sol_trend, tts_trend = pull_data()

    product_dc = {"ULY": dc_uly, "SOL": dc_sol, "TTS": dc_tts}
    product_trend = {"ULY": uly_trend, "SOL": sol_trend, "TTS": tts_trend}

    # same validation the Streamlit page runs before trusting the data --
    # a bad schema should fail the scheduled build loudly rather than ship
    # a broken dc_ocap.html. Warnings are printed but must not stop the
    # build: this runs hourly and uploads to S3, so failing over a few
    # wafers missing from trend would freeze the portal on a stale report.
    problems, warnings = check_data(product_dc, product_trend)
    if problems:
        raise SystemExit(
            "pull_data() 가 돌려준 데이터가 대시보드 형식과 맞지 않습니다:\n"
            + "\n".join(f"- {p}" for p in problems)
        )
    for w in warnings:
        print(f"  참고: {w}")

    data = {
        "dc": {p: _columns(product_dc[p]) for p in product_dc},
        "trend": {p: _columns(product_trend[p]) for p in product_trend},
        "itemCols": {
            p: [c for c in product_trend[p].columns if c not in META_TREND_COLS]
            for p in product_trend
        },
    }

    generated_at = datetime.now(KST).strftime("%y/%m/%d %H:%M")

    # separators=(",", ":"): json.dumps's default puts a space after every
    # comma and colon, which adds up across a few hundred thousand values
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    # gzip is what actually shrinks this (tabular numbers and repeated ids
    # compress extremely well); base64 is only the wrapper that lets the
    # compressed bytes sit inside a text/HTML file. base64 on its own would
    # make the payload ~33% BIGGER -- it's the pairing that wins. The page
    # inflates it with the browser's built-in DecompressionStream, so this
    # still needs no external library.
    encoded = base64.b64encode(gzip.compress(payload, 9)).decode("ascii")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    def fill(text: str, placeholder: str, value: str) -> str:
        """Substitute a placeholder, refusing to continue if it isn't there.

        str.replace() on a missing needle is a silent no-op, which is the
        worst possible outcome here: the build "succeeds" and writes an
        html file that simply has no data in it, and the failure only
        shows up later as a null-reference error in the browser. An
        out-of-date dc_ocap_template.html next to an updated app.py hits
        exactly this, so fail here instead.
        """
        if placeholder not in text:
            raise SystemExit(
                f"{TEMPLATE_PATH.name} 에서 '{placeholder}' 를 찾지 못했습니다.\n"
                f"app.py 와 {TEMPLATE_PATH.name} 의 버전이 서로 다른 것 같습니다 "
                f"(둘은 같은 커밋의 짝으로 써야 합니다).\n"
                f"두 파일을 함께 최신으로 받아서 다시 실행하세요."
            )
        return text.replace(placeholder, value)

    html = fill(template, "__GENERATED_AT__", generated_at)
    html = fill(html, "__DATA_B64__", encoded)
    # embedded rather than loaded from the public CDN: the portal server or
    # its viewers may not have outbound internet access, only reachability
    # to wherever this file itself gets hosted
    html = fill(html, "/*__PLOTLY_JS__*/", pyo.offline.get_plotlyjs())

    OUTPUT_PATH.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"wrote {OUTPUT_PATH} ({size_kb:.0f} KB)")
    print(f"  data: {len(payload)/1024:.0f} KB json -> {len(encoded)/1024:.0f} KB gzip+base64 "
          f"({len(encoded)/len(payload)*100:.0f}%)")
    return OUTPUT_PATH


# Three ways this file gets loaded, and what each one should do:
#
#   python app.py          -> build the static dc_ocap.html export
#   streamlit run app.py   -> render the dashboard standalone (preview)
#   import from portal.py  -> define show_dc_ocap() and nothing else
#
# __name__ alone can't tell the first two apart: `streamlit run` also
# executes the script as "__main__", so guarding on that by itself would
# silently re-run the whole export -- a full data pull and a multi-MB file
# write -- on every widget click, while rendering a blank page because
# nothing called show_dc_ocap(). st.runtime.exists() is what separates
# them; it's False under a plain interpreter and True inside a running
# Streamlit server. The import case is already excluded by __name__.
if __name__ == "__main__":
    if st.runtime.exists():
        # standalone preview: this file is the main script, so nothing else
        # has claimed set_page_config yet (portal.py calls its own)
        st.set_page_config(page_title="DC OCAP", layout="wide")
        show_dc_ocap()
    else:
        build_dc_ocap_html()

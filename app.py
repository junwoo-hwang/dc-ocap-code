"""OCAP hold dashboard.

Left: the selected product's hold list (uly_dc / sol_dc / tts_dc),
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
it only has to end up with pull_data() returning these twelve dataframes,
in this order: uly_dc / sol_dc / tts_dc, uly_trend / sol_trend /
tts_trend, uly_spec / sol_spec / tts_spec, uly_split / sol_split /
tts_split. The order is the contract -- the dashboard reads them by
position, not by name.

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

# split(EIN/ECN 적용 이력) 용. dc(hold) 와는 다른 계통이라 lot 만 겹칠 뿐
# 같은 건을 가리키지 않는다. 원본에서는 comp_id_list 한 칸에
# "ABABC.04,ABABC.13,..." 처럼 들어오는데, 여기 mock 은 이미 wafer 번호
# 칸(1~25)으로 펼치고 같은 건끼리 합친 뒤의 모양으로 만든다.
EIN_ECN_TYPES = ["EIN", "ECN"]
SPLIT_STEP_DESCS = [
    "PHOTO ALIGN KEY 재설정",
    "ETCH CHAMBER SEASONING 조건 변경",
    "CMP PAD 교체 후 조건 재설정",
    "IMPLANT DOSE 보정",
    "CLEAN RECIPE STEP 추가",
    "METAL DEPO TARGET 교체",
    "ANNEAL TEMP PROFILE 변경",
]
SPLIT_TITLES = [
    "설비 PM 후 조건 재적용",
    "신규 recipe 적용 평가",
    "수율 개선 조건 split 평가",
    "계측 산포 개선 조건 확인",
    "대체 설비 적용 평가",
    "원자재 lot 변경 검증",
]
SPLIT_REASONS = [
    "PM 이후 첫 적용분 확인 필요",
    "직전 lot 산포 확대 대응",
    "고객 요청 조건 변경",
    "설비 alarm 이력 연계 확인",
    "양산 적용 전 소량 평가",
]
SPLIT_N_WAFER = 25
SPLIT_WAFER_COLS = [str(n) for n in range(1, SPLIT_N_WAFER + 1)]

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


# 규격이 한 번 상향된 날. 이 앞뒤로 값이 달라져야 차트에 계단이 생긴다.
SPEC_CHANGE = datetime(2026, 8, 1)


def generate_spec_for_product(product: str, trend_df: pd.DataFrame) -> pd.DataFrame:
    """Mock spec table: one row per (item_id, from_time) revision.

    Keyed by when the spec took effect, which is what a spec actually is --
    a revision with a start date. It stays a few dozen rows however much
    history accumulates, and the limit line it draws is two or three points
    that step exactly on the revision date.
    """
    cfg = PRODUCT_CONFIG[product]
    rng = np.random.default_rng(cfg["seed"] + 7)
    start = trend_df["tkout_time"].min() if not trend_df.empty else datetime(2026, 1, 1)

    rows = []
    for item in item_columns(trend_df):
        spread, center = cfg["spread"], cfg["center"]
        first = {
            "usl": round(center + rng.uniform(spread * 1.5, spread * 2.5), 3),
            "lsl": round(center - rng.uniform(spread * 1.5, spread * 2.5), 3),
            "ucl": round(center + rng.uniform(spread * 0.6, spread * 1.2), 3),
            "lcl": round(center - rng.uniform(spread * 0.6, spread * 1.2), 3),
        }
        rows.append({"item_id": item, "from_time": start, **first})
        # 일부 item 만 바꾼다: 계단도, 평평한 선도 둘 다 시험해야 한다
        if rng.random() < 0.4:
            bump = round(rng.uniform(spread * 0.3, spread * 0.8), 3)
            rows.append({"item_id": item, "from_time": SPEC_CHANGE,
                         **{**first, "usl": round(first["usl"] + bump, 3),
                            "ucl": round(first["ucl"] + bump, 3)}})
    return (pd.DataFrame(rows)[SPEC_REQUIRED]
            .sort_values(["item_id", "from_time"]).reset_index(drop=True))


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
    # item_columns(): item1 은 고르고 item1_usl 은 거른다. startswith("item")
    # 으로 잡으면 관리선 컬럼이 hold 대상 item 으로 섞여 들어간다
    item_cols = item_columns(trend_df)


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


def generate_split_for_product(product: str, trend_df: pd.DataFrame,
                               dc_df: pd.DataFrame | None = None,
                               n_rows: int = 40, seed: int | None = None) -> pd.DataFrame:
    """Generate a mock EIN/ECN split dataframe for a single product.

    Columns, in this order:
      einecn_no, root_lot_id, ppid, ein_ecn_type, "1".."25",
      step_seq, step_desc, title, reason, process_id

    "1".."25" are wafer numbers: "V" if that wafer is in the split, "" if
    not. This is the shape *after* comp_id_list has been spread out and
    rows that only differed by comp_id_list have been merged, which is how
    the datalake hands it over -- so one (einecn_no, root_lot_id, ppid,
    ein_ecn_type, step_seq) never appears twice.

    Two things are deliberately left blank, because they are blank in the
    real data and anything reading this has to cope:
      - reason: some rows have None (a datalake NULL) and some have "".
        Both occur; code that only checks one of them will miss the other.
      - the wafer columns: most are "" on any given row. Every row has at
        least one "V" though -- a split that touched no wafer is dropped
        upstream, so it never reaches here.

    root_lot_id and the wafer numbers are sampled from `trend_df`, so a
    split can actually be looked up against the measurements.

    Pass `dc_df` to weight the lots towards ones that were actually held.
    Without it the lots are drawn flat and most held lots end up with no
    EIN/ECN history at all, so the EINECN button on the dashboard has
    nothing to show on almost every hold -- which is not what the real
    data looks like, and leaves the popup untested.
    """
    rng = np.random.default_rng(seed)
    cfg = PRODUCT_CONFIG[product]
    # 정렬해 둔다 -- unique() 는 등장 순서라 위쪽 mock 이 바뀌면 같이 흔들린다
    lots = sorted(trend_df["root_lot_id"].unique())
    held_lots = []
    if isinstance(dc_df, pd.DataFrame) and "root_lot_id" in dc_df.columns:
        held_lots = sorted(set(dc_df["root_lot_id"]) & set(lots))
    wafers_by_lot = {lot: sorted(int(w) for w in g["wafer_id"].unique())
                     for lot, g in trend_df.groupby("root_lot_id")}
    # 제품 하나가 쓰는 ppid 는 몇 개뿐이라 풀에서 골라 쓴다
    ppids = ["".join(rng.choice(LOT_ID_CHARS, size=int(rng.integers(12, 23))))
             for _ in range(4)]

    rows = []
    while len(rows) < n_rows:
        einecn_no = (
            "".join(rng.choice(list(string.ascii_uppercase), size=3))
            + f"{rng.integers(0, 1000000):06d}"
            + "".join(rng.choice(LOT_ID_CHARS, size=3))
            + f"-{rng.integers(0, 3)}"
        )
        ein_ecn_type = str(rng.choice(EIN_ECN_TYPES, p=[0.75, 0.25]))
        title = str(rng.choice(SPLIT_TITLES))
        # reason 은 title 과 같이 '그 test 를 왜 했나' 라서 test 단위로 붙는다
        # (팝업에서도 einecn_no 에 마우스를 올리면 둘이 같이 뜬다)
        draw = rng.random()
        reason = None if draw < 0.2 else ("" if draw < 0.35
                                          else str(rng.choice(SPLIT_REASONS)))

        # 하나의 test(einecn_no)를 step 여럿에 묶어서 돌린다. 그러니 lot 을
        # 먼저 하나 정하고, 그 lot 안에서 step 을 여러 개 뽑는다 -- step 마다
        # 한 줄이고, 적용된 wafer 도 step 마다 다르다.
        # hold 가 걸린 lot 쪽으로 기울여 뽑는다 (위 docstring 참고)
        lot_pool = (held_lots if held_lots and rng.random() < 0.7 else lots)
        # 가끔 한 test 가 lot 두 개에 걸친다
        n_lots = 2 if (len(lot_pool) > 1 and rng.random() < 0.2) else 1
        for lot in rng.choice(lot_pool, size=n_lots, replace=False):
            lot = str(lot)
            # 같은 (lot, step) 이 두 번 나오면 합쳐졌어야 할 행이 두 줄로
            # 남는다. step 을 겹치지 않게 뽑는다.
            steps = set()
            while len(steps) < int(rng.integers(1, 4)):
                steps.add("".join(rng.choice(list(string.ascii_uppercase), size=2))
                          + f"{rng.integers(0, 1000000):06d}")

            for step_seq in sorted(steps):
                step_desc = str(rng.choice(SPLIT_STEP_DESCS))
                # 같은 step 이 ppid 여러 개로 나뉘어 오기도 한다. 팝업은
                # step_seq / step_desc 칸을 합쳐 그리므로, 그 경우가 mock 에
                # 없으면 병합이 한 번도 안 그려져 확인이 안 된다.
                # 한 step 이 여러 줄로 나뉘어 오고, 줄마다 wafer 가 다르다.
                # ppid 는 겹칠 수 있게 뽑는다(replace=True): 나누는 기준이
                # 설비인 경우가 있는데 설비 칸은 안 받아오므로, 그런 줄들은
                # comp_id_list 말고는 완전히 같은 모습으로 들어온다.
                n_group = 1 if rng.random() < 0.6 else int(rng.integers(2, 4))
                step_ppids = rng.choice(ppids, size=n_group, replace=True)
                # 한 wafer 는 한 갈래에만 들어간다 -- 갈래별 wafer 는 겹치지 않는다
                pool = wafers_by_lot[lot]
                taken = rng.permutation(pool)[:min(len(pool), int(rng.integers(1, 14)))]
                shares = np.array_split(taken, len(step_ppids))

                for step_ppid, share in zip(step_ppids, shares):
                    hit = set(int(w) for w in share)
                    if not hit:
                        continue      # wafer 가 없는 줄은 앞 단계에서 걸러진다
                    row = {
                        "einecn_no": einecn_no,
                        "root_lot_id": lot,
                        "ppid": str(step_ppid),
                        "ein_ecn_type": ein_ecn_type,
                        "step_seq": step_seq,
                        "step_desc": step_desc,
                        "title": title,
                        "reason": reason,
                        "process_id": cfg["process_id"],
                    }
                    for n in range(1, SPLIT_N_WAFER + 1):
                        row[str(n)] = "V" if n in hit else ""
                    rows.append(row)

    return pd.DataFrame(rows)[
        ["einecn_no", "root_lot_id", "ppid", "ein_ecn_type"]
        + SPLIT_WAFER_COLS
        + ["step_seq", "step_desc", "title", "reason", "process_id"]
    ]


def pull_data():
    """Return dc x3, trend x3, spec x3, split x3 (ULY, SOL, TTS each time).

    spec_* holds the control limits as revisions --
    (item_id, from_time, ucl, lcl, usl, lsl), one row per time the spec
    changed. They are not in dc because the spec in force changes over
    time: a wafer measured in July and one measured in August are judged
    against different numbers, and the chart steps on the date it changed.

    split_* is the EIN/ECN application history, one frame per product --
    in the real pull, one `split` table sliced on process_id
    (KNNU=ULY, KNJO=SOL, KNIK=TTS). Mind the order below: this function
    returns ULY, SOL, TTS, so slicing them out in a different order and
    returning them as written would quietly label one product's splits
    with another's name.

    Put the real company-system pull in here. It must be a function, not
    bare module-level code: Streamlit re-runs this file top to bottom on
    every click, so anything at module level would be re-fetched on every
    row selection. The dashboard below calls this through a cache.
    """
    uly_trend = generate_probe_df("ULY")
    sol_trend = generate_probe_df("SOL")
    tts_trend = generate_probe_df("TTS")

    uly_dc = generate_dc_for_product("ULY", uly_trend, n_rows=50, seed=201)
    sol_dc = generate_dc_for_product("SOL", sol_trend, n_rows=50, seed=202)
    tts_dc = generate_dc_for_product("TTS", tts_trend, n_rows=50, seed=203)

    uly_spec = generate_spec_for_product("ULY", uly_trend)
    sol_spec = generate_spec_for_product("SOL", sol_trend)
    tts_spec = generate_spec_for_product("TTS", tts_trend)

    uly_split = generate_split_for_product("ULY", uly_trend, uly_dc, seed=301)
    sol_split = generate_split_for_product("SOL", sol_trend, sol_dc, seed=302)
    tts_split = generate_split_for_product("TTS", tts_trend, tts_dc, seed=303)

    return (uly_dc, sol_dc, tts_dc,
            uly_trend, sol_trend, tts_trend,
            uly_spec, sol_spec, tts_spec,
            uly_split, sol_split, tts_split)


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

import numpy as np
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
    "hold_inform", "line_id", "process_id",
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


# Limits live in trend next to the measurement, as item1_ucl / item1_usl /
# ... beside item1, because the spec in force changes over time and each row
# has to carry the one that applied when that wafer came out of test.
LIMIT_COLS = ("ucl", "lcl", "usl", "lsl")
SPEC_REQUIRED = ["item_id", "from_time", *LIMIT_COLS]
# trend's own columns, everything else is either an item or one of its limits
META_TREND_COLS = ["root_lot_id", "wafer_id", "tkout_time",
                   "probe_card_id", "eqp_id", "lot_type", "rw_cnt"]

# EIN/ECN 적용 이력(split) 에서 EINECN 팝업이 읽는 것들.
# "1".."25" 는 wafer 번호 칸이고, 그 wafer 가 적용 대상이면 "V" 가 들어 있다.
SPLIT_WAFER_COLUMNS = [str(n) for n in range(1, 26)]
# 팝업 표의 칸 순서 그대로다. root_lot_id 는 어느 lot 의 이력인지 찾는 데
# 쓰고, title 은 einecn_no 에 마우스를 올렸을 때 뜬다 -- 둘 다 표에는 없다.
SPLIT_REQUIRED = ["root_lot_id", "title", "einecn_no", "step_seq", "step_desc",
                  "ppid", "reason", "ein_ecn_type", *SPLIT_WAFER_COLUMNS]


def item_columns(trend_df) -> list:
    """The measurement columns of a trend frame (everything but its metadata)."""
    return [c for c in trend_df.columns if str(c) not in META_TREND_COLS]


def item_spec_rows(spec_df, item_id) -> pd.DataFrame:
    """One item's revisions, oldest first, with the limits coerced to float."""
    empty = pd.DataFrame(columns=SPEC_REQUIRED)
    if not isinstance(spec_df, pd.DataFrame) or spec_df.empty:
        return empty
    if "item_id" not in spec_df.columns or "from_time" not in spec_df.columns:
        return empty
    key = str(item_id).strip().lower()
    rows = spec_df[spec_df["item_id"].astype(str).str.strip().str.lower() == key].copy()
    if rows.empty:
        return rows
    rows["from_time"] = pd.to_datetime(rows["from_time"], errors="coerce")
    for w in LIMIT_COLS:
        rows[w] = pd.to_numeric(rows[w], errors="coerce") if w in rows.columns else np.nan
    # 관리선 값까지 포함해 정렬한다. 같은 item 이 같은 from_time 에 두 벌
    # 들어오는 일이 실제로 있는데(뽑는 쿼리에 조건이 하나 모자란 경우),
    # from_time 만으로 정렬하면 '나중 행' 이 원본 행 순서에 따라 달라져서
    # 같은 데이터인데 조회할 때마다 관리선이 달라 보인다. 두 벌 중 어느
    # 쪽이 맞는지는 여기서 알 수 없으므로 고르지는 않고, 적어도 항상 같은
    # 쪽이 고르도록만 해 둔다 (진짜 원인은 diagnose.py 7번이 짚어준다).
    return (rows.dropna(subset=["from_time"])
            .sort_values(["from_time", *LIMIT_COLS], na_position="first"))


def limits_asof(spec_rows: pd.DataFrame, times) -> dict:
    """Limits in force at each of `times` -> {which: Series aligned to times}.

    merge_asof(direction="backward") is exactly "which revision was live
    when this was measured": the newest revision at or before each
    measurement, and NaN for anything measured before the first one.
    """
    idx = getattr(times, "index", None)
    t = pd.Series(pd.to_datetime(pd.Series(times).values, errors="coerce"))
    if spec_rows.empty:
        out = {w: pd.Series(np.nan, index=t.index, dtype="float64") for w in LIMIT_COLS}
    else:
        left = pd.DataFrame({"_t": t}).reset_index()
        merged = (pd.merge_asof(left.sort_values("_t"),
                                spec_rows[["from_time", *LIMIT_COLS]],
                                left_on="_t", right_on="from_time",
                                direction="backward")
                  .set_index("index").reindex(t.index))
        out = {w: merged[w].astype("float64") for w in LIMIT_COLS}
    if idx is not None:
        for w in out:
            out[w].index = idx
    return out


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


def check_data(product_dc: dict, trend_frames: dict,
               spec_frames: dict | None = None) -> tuple[list[str], list[str]]:
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
            (f"{product.lower()}_dc", dc_df, DC_REQUIRED),
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
            (f"{product.lower()}_dc", dc_df, "hold_time"),
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

        # dc 가 가리키는 wafer 가 trend 에 실제로 있는가. 이게 어긋나면 hold
        # 가 차트에 빨간 점으로 안 찍히는데, 화면에는 아무 오류도 안 뜬다.
        # dtype 이 서로 달라도 norm_lot/norm_wafer 가 흡수하므로, 정규화한
        # 쌍이 겹치는지만 본다.
        pair_cols = {"root_lot_id", "wafer_id"}
        if pair_cols <= set(dc_df.columns) and pair_cols <= set(trend_df.columns) and not dc_df.empty:
            trend_pairs = set(zip(trend_df["root_lot_id"].map(norm_lot),
                                  trend_df["wafer_id"].map(norm_wafer)))
            dc_pairs = set(zip(dc_df["root_lot_id"].map(norm_lot),
                               dc_df["wafer_id"].map(norm_wafer)))
            missing = dc_pairs - trend_pairs
            if missing and len(missing) == len(dc_pairs):
                # 하나도 안 맞으면 배선이 틀린 것이다 (다른 컬럼을 뽑았거나
                # 제품을 잘못 짝지었거나). 이건 막아야 한다.
                problems.append(
                    f"{product.lower()}_dc: (root_lot_id, wafer_id) 가 "
                    f"{product.lower()}_trend 에서 하나도 매칭되지 않습니다. "
                    f"dc 예시 {sorted(missing)[:3]}"
                )
            elif missing:
                # 일부만 없는 것: 그 wafer 만 빨간 점이 안 찍히고 나머지는
                # 정상이다. bad_pairs 는 단순 집합 조회라 죽지도 않는다.
                # 흔한 원인은 dc 와 trend 의 조회 기간 기준이 다른 것
                # (dc=hold_time, trend=tkout_time) 이다.
                warnings.append(
                    f"{product.lower()}_dc: {len(missing)}/{len(dc_pairs)} 건의 "
                    f"(root_lot_id, wafer_id) 가 {product.lower()}_trend 에 없습니다 "
                    f"(해당 hold 는 빨간 점이 안 찍힘). 예시 {sorted(missing)[:3]}"
                )

        # ucl/lcl/usl/lsl must be numeric: plotly's add_hline raises a
        # TypeError deep inside its own layout code (not a clean KeyError)
        # if one of these came back as text (e.g. a BigQuery NUMERIC that
        # round-tripped as str/Decimal). Cheap to check in full -- only 4
        # columns, unlike the per-item trend checks below.
        # 관리선은 spec 프레임에 (item_id, from_time) 개정 이력으로 있다.
        # 없거나 안 맞아도 그 점이 회색으로 그려질 뿐이라 대개 경고다.
        spec_df = (spec_frames or {}).get(product)
        if not isinstance(spec_df, pd.DataFrame):
            warnings.append(
                f"{product.lower()}_spec: DataFrame 이 아닙니다 "
                f"({type(spec_df).__name__}). 관리선 없이 그려집니다."
            )
        elif spec_df.empty:
            warnings.append(f"{product.lower()}_spec: 비어 있습니다. 관리선 없이 그려집니다.")
        else:
            missing = [c for c in SPEC_REQUIRED if c not in spec_df.columns]
            if missing:
                problems.append(f"{product.lower()}_spec: 컬럼 없음 -> {', '.join(missing)}")
            else:
                if not pd.api.types.is_datetime64_any_dtype(spec_df["from_time"]):
                    n_bad = pd.to_datetime(spec_df["from_time"], errors="coerce").isna().sum()
                    warnings.append(
                        f"{product.lower()}_spec: from_time 이 날짜형이 아닙니다 "
                        f"({spec_df['from_time'].dtype})"
                        + (f", 그 중 {n_bad}개는 날짜로 읽히지도 않습니다" if n_bad else "")
                        + ". pd.to_datetime() 으로 변환하세요 "
                        "(변환은 자동으로도 하지만, 읽히지 않는 값은 그 개정이 통째로 빠집니다)."
                    )
                bad_lim = [f"{c}({n}개)" for c in LIMIT_COLS
                           if (n := spec_df[c].map(
                               lambda v: to_float(v) is None and pd.notna(v)).sum())]
                if bad_lim:
                    warnings.append(
                        f"{product.lower()}_spec: 관리선 값이 숫자가 아닙니다 -> {bad_lim[:4]}. "
                        f"해당 선은 안 그려집니다. float 로 변환하세요."
                    )
                # trend 의 item 이 spec 에 있는가
                if not trend_df.empty:
                    have = set(spec_df["item_id"].astype(str).str.strip().str.lower())
                    items = [str(c) for c in item_columns(trend_df)]
                    miss = [i for i in items if i.strip().lower() not in have]
                    if miss and len(miss) == len(items):
                        problems.append(
                            f"{product.lower()}_spec: trend 의 item 이 하나도 매칭되지 "
                            f"않습니다. trend 예시 {items[:4]}, spec 예시 {sorted(have)[:4]}"
                        )
                    elif miss:
                        warnings.append(
                            f"{product.lower()}_spec: item {miss[:5]} 이(가) 없습니다 "
                            f"({len(miss)}/{len(items)}종). 해당 차트는 관리선 없이 회색으로만 그려집니다."
                        )

    return fatal, warnings


def frames_by_product(frames) -> tuple[dict, dict, dict, dict]:
    """pull_data() 의 12개 튜플 -> (dc, trend, spec, split) 제품별 dict 4개.

    순서로 받은 걸 이름으로 바꾸는 자리는 여기 하나뿐이다. 여러 군데서
    각자 풀면 한 곳만 순서를 잘못 적어도 조용히 다른 제품 데이터를 그리게
    된다. load_data() 가 뒤에 붙이는 loaded_at/problems/warnings 도 그대로
    넘길 수 있도록 남는 건 무시한다.

    split 은 아직 화면에서 읽는 데가 없다. 그래도 여기서 같이 이름을
    붙여 두는 이유는, 나중에 쓸 때 다른 데서 순서로 풀지 않게 하려는 것이다.
    """
    (uly_dc, sol_dc, tts_dc, uly_trend, sol_trend, tts_trend,
     uly_spec, sol_spec, tts_spec, uly_split, sol_split, tts_split,
     *_rest) = frames
    # 순서는 화면의 제품 전환 버튼과 같게 둔다 (ULY / TTS / SOL)
    return ({"ULY": uly_dc, "TTS": tts_dc, "SOL": sol_dc},
            {"ULY": uly_trend, "TTS": tts_trend, "SOL": sol_trend},
            {"ULY": uly_spec, "TTS": tts_spec, "SOL": sol_spec},
            {"ULY": uly_split, "TTS": tts_split, "SOL": sol_split})


# Streamlit re-runs show_dc_ocap() on every click (the portal reruns its
# whole script top to bottom, same as any Streamlit app), so pull_data()
# is called through a cache -- otherwise every row selection would
# re-query the company system. ttl is how stale the data may get before
# the next interaction refetches it; raise or lower it to taste, and use
# the app's ⋮ menu > Clear cache to force a refresh.
@st.cache_data(ttl=600, show_spinner="데이터 불러오는 중...")
def load_data():
    frames = pull_data()
    # checked here rather than on every rerun: it scans the whole trend
    # tables, which is far too slow to repeat on each click
    dc_frames, trend_frames, spec_frames, _split_frames = frames_by_product(frames)
    problems, warnings = check_data(dc_frames, trend_frames, spec_frames)
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


LIMIT_LINE_STYLE = {
    "ucl": ("blue", "UCL"), "lcl": ("blue", "LCL"),
    "usl": ("red", "USL"),  "lsl": ("red", "LSL"),
}


def add_limit_steps(fig, spec_rows: pd.DataFrame, x_min, x_max) -> None:
    """Draw each limit as a step line built from the spec revisions.

    Taken from the revisions rather than the measurements, so the step
    lands on the date the spec changed instead of on the first wafer
    measured after it, the line spans the chart even across a gap in
    measurements, and it costs two or three points instead of one per row.
    """
    if spec_rows.empty or pd.isna(x_min) or pd.isna(x_max):
        return
    for which in LIMIT_COLS:
        xs, ys = [], []
        for t, v in zip(spec_rows["from_time"], spec_rows[which]):
            if pd.isna(v):
                continue
            if t <= x_min:
                # 차트 시작 시점에 이미 적용 중이던 값 -- 왼쪽 끝에서 시작
                xs, ys = [x_min], [v]
            elif t <= x_max:
                xs.append(t)
                ys.append(v)
        if not xs:
            continue
        xs.append(x_max)          # 마지막 값을 오른쪽 끝까지 끌고 간다
        ys.append(ys[-1])
        color, label = LIMIT_LINE_STYLE[which]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=label,
            # hv: hold the old value until the moment it changes, then step
            line=dict(color=color, dash="dash", width=1, shape="hv"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_annotation(
            xref="paper", x=0, xanchor="left", y=ys[0], yref="y", text=label,
            showarrow=False, font=dict(color=color, size=11),
            yanchor="bottom" if which in ("ucl", "usl") else "top",
        )


def build_scatter(trend_df, item_id: str, bad_pairs: set, bad_label: str,
                   legend_field: str | None, spec_df,
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
    # coerce before dropna: a numeric column that came back as text/object
    # (BigQuery NUMERIC/DECIMAL) would otherwise keep its non-null string
    # values and only fail once plotly tries to lay out the chart
    plot_df = trend_df.assign(**{item_id: pd.to_numeric(trend_df[item_id], errors="coerce")})
    plot_df = plot_df.dropna(subset=[item_id])
    # keep tkout_time itself as a real datetime (needed for the x-axis);
    # format a separate string column just for the hover box
    plot_df["_hover_time"] = plot_df["tkout_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # limits come from the row, not from one number for the whole chart: the
    # spec in force changes over time, so each point is judged against the
    # one that applied when it was measured
    spec_rows = item_spec_rows(spec_df, item_id)
    lim = limits_asof(spec_rows, plot_df["tkout_time"])
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
    past_scrap = (values > lim["usl"]).fillna(False) | (values < lim["lsl"]).fillna(False)
    past_control = ((values > lim["ucl"]).fillna(False)
                    | (values < lim["lcl"]).fillna(False))
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
    add_limit_steps(fig, spec_rows,
                    plot_df["tkout_time"].min(), plot_df["tkout_time"].max())

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
# WAC Trend 페이지 -- 한 제품의 item 을 전부 작은 차트로 늘어놓고, 어느
# 차트에서 타점을 누르든 그 wafer 가 모든 차트에서 같이 커진다.
# 정적 리포트(dc_ocap_template.html)의 WAC 페이지와 같은 규칙/색/크기를
# 쓴다 -- 값이 갈리면 두 화면이 다른 판정을 내리게 된다.
# ====================================================================
WAC_GRAY, WAC_SAME = "lightgray", "dimgray"
WAC_SIZE, WAC_SIZE_SEL = 6, 14
# 배경(회색) 타점이 이보다 많으면 고르게 솎는다. 260px 차트에 수천 점을
# 찍어봐야 대부분 겹치고, 브라우저로 넘길 JSON 만 그만큼 커진다.
# CL OUT / SL OUT 은 절대 솎지 않는다 -- 그게 봐야 할 신호다.
WAC_MAX_GRAY = 2500
WAC_CHART_HEIGHT = 260
WAC_GRID_COLS = 2
# grp 값 -> (범례 이름, 색, 범례 순서). 0=정상 1=CL OUT 2=SL OUT
WAC_GROUPS = (
    ("trend", WAC_GRAY, 1100),
    ("CL OUT", LIMIT_COLORS["control"], 20),
    ("SL OUT", LIMIT_COLORS["scrap"], 10),
)


@st.cache_data(show_spinner=False)
def wac_item_points(product: str, item_col: str, stamp: str) -> dict:
    """한 (제품, item) 의 타점을 그리기 좋은 배열로 미리 만들어 캐시한다.

    타점을 한 번 누를 때마다 Streamlit 은 페이지를 통째로 다시 그린다.
    item 이 30개면 그때마다 30번 * 수천 행을 다시 훑게 되므로, 판정(CL/SL)
    과 hover 문자열까지 여기서 한 번만 만들어 둔다. 선택 표시는 이 배열 위의
    boolean mask 라서 클릭할 때 다시 계산할 게 거의 없다.

    stamp 는 load_data() 가 찍은 적재 시각이다. 데이터프레임을 인자로 받으면
    Streamlit 이 캐시 키를 만들려고 매번 전체를 해시하는데, 그게 계산보다
    비싸다. 대신 값싼 문자열을 키로 쓰고 프레임은 (이미 캐시된) load_data()
    에서 가져온다 -- 데이터가 새로 적재되면 stamp 가 바뀌어 같이 무효화된다.
    """
    _dc, trend_frames, spec_frames, _split = frames_by_product(load_data())
    trend_df, spec_df = trend_frames[product], spec_frames[product]

    vals = pd.to_numeric(trend_df[item_col], errors="coerce")
    keep = vals.notna()
    d, v = trend_df[keep], vals[keep]
    spec_rows = item_spec_rows(spec_df, item_col)
    lim = limits_asof(spec_rows, d["tkout_time"])

    # 규격을 벗어난 점이 관리선도 벗어난 건 당연하므로, SL 이 이기고 둘은
    # 서로 겹치지 않는다 (build_scatter 와 같은 규칙)
    scrap = (v > lim["usl"]).fillna(False) | (v < lim["lsl"]).fillna(False)
    control = ((v > lim["ucl"]).fillna(False)
               | (v < lim["lcl"]).fillna(False)) & ~scrap
    grp = np.where(scrap, 2, np.where(control, 1, 0)).astype(np.int8)

    hover = d["tkout_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cd = np.column_stack([
        d["root_lot_id"].astype(str), d["wafer_id"].astype(str), hover,
        d["probe_card_id"].astype(str), d["eqp_id"].astype(str),
        d["lot_type"].astype(str), d["rw_cnt"].astype(str),
    ])

    gray = np.flatnonzero(grp == 0)
    if WAC_MAX_GRAY and len(gray) > WAC_MAX_GRAY:
        step = len(gray) / WAC_MAX_GRAY
        pick = np.floor(np.arange(WAC_MAX_GRAY) * step).astype(int)
        # 마지막 점은 따로 넣는다. floor 는 끝에 못 닿아서 그냥 두면 차트가
        # 실제보다 일찍 끝난 것처럼 보인다 (정적 리포트도 같이 맞춰 뒀다)
        if pick[-1] != len(gray) - 1:
            pick = np.append(pick, len(gray) - 1)
        gray = gray[pick]

    x = d["tkout_time"].to_numpy()
    return {
        "x": x, "y": v.to_numpy(), "cd": cd, "grp": grp, "gray": gray,
        "lot": d["root_lot_id"].map(norm_lot).to_numpy(),
        "wafer": d["wafer_id"].map(norm_wafer).to_numpy(),
        "spec_rows": spec_rows,
        "x_min": x.min() if len(x) else None,
        "x_max": x.max() if len(x) else None,
    }


def build_wac_scatter(pts: dict, item_col: str, selection: dict | None) -> go.Figure:
    """WAC 그리드의 차트 하나. 선택된 wafer 는 모든 차트에서 같이 커진다."""
    fig = go.Figure()
    x, y, cd, grp = pts["x"], pts["y"], pts["cd"], pts["grp"]

    for gi, (name, color, rank) in enumerate(WAC_GROUPS):
        idx = pts["gray"] if gi == 0 else np.flatnonzero(grp == gi)
        # 비어도 트레이스는 넣는다 -- 범례에 CL OUT / SL OUT 자리가 항상
        # 있어야 "없는 것" 과 "안 그려진 것" 을 구분할 수 있다
        fig.add_trace(go.Scatter(
            x=x[idx], y=y[idx], mode="markers", name=name, legendrank=rank,
            marker=dict(color=color, size=WAC_SIZE),
            customdata=cd[idx], hovertemplate=HOVER_TEMPLATE,
        ))

    # 선택 표시는 바탕 위에 얹는 '덧그림' 두 개다. 선택이 없어도 빈 채로
    # 넣어 트레이스 자리와 순서를 고정한다 -- 정적 리포트가 덧그림을 3, 4번
    # 자리에 못 박아 두고 restyle 만 하는 것과 같은 구성이라, 한쪽을 고칠 때
    # 다른 쪽에서 무엇을 고쳐야 하는지가 바로 보인다.
    lot_gray = np.array([], dtype=int)
    hit_idx = np.array([], dtype=int)
    if selection:
        same_lot = pts["lot"] == norm_lot(selection["root_lot_id"])
        hit = same_lot & (pts["wafer"] == norm_wafer(selection["wafer_id"]))
        # 같은 lot 은 회색 타점만 진하게. CL/SL 은 제 색을 잃으면 안 된다
        lot_gray = np.flatnonzero(same_lot & ~hit & (grp == 0))
        hit_idx = np.flatnonzero(hit)
    # hoverinfo="skip" 은 이 트레이스의 클릭도 끈다 -- 덧그림을 눌러도
    # 밑의 원래 타점이 잡히므로 클릭 처리가 한 곳으로 모인다
    fig.add_trace(go.Scatter(
        x=x[lot_gray], y=y[lot_gray], mode="markers",
        marker=dict(color=WAC_SAME, size=WAC_SIZE),
        showlegend=False, hoverinfo="skip",
    ))
    # 고른 타점: 커지되 색은 자기 그룹 색 그대로 -- CL/SL 인지 계속 보인다
    fig.add_trace(go.Scatter(
        x=x[hit_idx], y=y[hit_idx], mode="markers",
        marker=dict(color=np.array([g[1] for g in WAC_GROUPS])[grp[hit_idx]],
                    size=WAC_SIZE_SEL),
        showlegend=False, hoverinfo="skip",
    ))

    add_limit_steps(fig, pts["spec_rows"], pts["x_min"], pts["x_max"])
    fig.update_layout(
        xaxis_title="tkout_time", yaxis_title=item_col,
        legend=dict(font=dict(size=9), itemsizing="constant", tracegroupgap=0),
        height=WAC_CHART_HEIGHT, margin=dict(t=24, b=34, l=52, r=12),
        # 타점을 눌러 다시 그려도 확대/이동 상태는 그대로 둔다
        uirevision=item_col,
    )
    return fig


# 두 페이지의 h1 옆에 붙는 작은 회색 글씨. 0.42em 이라 h1 크기에 따라
# 같이 줄고, streamlit 이 h1 을 몇 px 로 그리든 두 페이지가 같아 보인다.
HEAD_META_STYLE = "font-size:0.42em; font-weight:400; color:#888; line-height:1.35;"
PAGES = ["DC OCAP", "WAC Trend"]


def clicked_customdata(chart_state):
    """st.plotly_chart 위젯 상태 -> 눌린 타점의 customdata (없으면 None).

    st.session_state[key] 로 미리 읽든, st.plotly_chart 가 돌려준 값을 쓰든
    같은 물건이라 한 함수로 처리한다. 모양이 바뀌어도 죽지 않게 방어적으로
    꺼낸다 -- 여기서 죽으면 화면 전체가 안 뜬다.
    """
    if not chart_state:
        return None
    selection = chart_state.get("selection") if hasattr(chart_state, "get") else None
    points = selection.get("points") if hasattr(selection, "get") else None
    if not points:
        return None
    first = points[0]
    cd = first.get("customdata") if hasattr(first, "get") else None
    return cd if cd else None


def render_wac_page(product_dc: dict, trend_frames: dict, spec_frames: dict,
                    data_loaded_at: str) -> None:
    """제품 하나의 item 을 전부 작은 차트로 늘어놓는 화면.

    타점을 누르면 그 wafer 가 모든 차트에서 같이 커지고, 같은 lot 의 회색
    타점은 진한 회색이 된다. 정적 리포트의 WAC 페이지와 같은 화면이다.

    정적 리포트는 스크롤에 맞춰 차트를 하나씩 그리지만(IntersectionObserver),
    Streamlit 은 파이썬이 그린 그림을 통째로 넘기는 구조라 그럴 자리가 없다.
    그래서 여기서는 한 번에 그릴 차트 수를 나눠서 넘긴다 -- 30개를 한꺼번에
    넘기면 클릭 한 번에 수 MB 를 다시 실어보내게 된다.
    """
    products = list(product_dc.keys())
    sel_key = f"{KEY_PREFIX}wac_selected"
    counts = ", ".join(f"{p} : {len(item_columns(trend_frames[p]))}건" for p in products)
    st.markdown(
        f"# WAC Trend "
        f"<span style='display:inline-block; vertical-align:bottom; {HEAD_META_STYLE}'>"
        f"WAC item 수<br>{counts}</span>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([2, 3])
    with left:
        title_col, search_col = st.columns([1.3, 3.35])
        with title_col:
            st.markdown(
                "<div style='font-size:1.15rem; font-weight:700; padding-top:0.3rem;'>"
                "ITEM Trend</div>",
                unsafe_allow_html=True,
            )
        with search_col:
            search = st.text_input(
                "item 검색", placeholder="item_id 검색 (예: item3)",
                label_visibility="collapsed", key=f"{KEY_PREFIX}wac_search",
            )
        # DC OCAP 페이지와 같은 key 를 쓴다 -- 정적 리포트처럼 두 화면이
        # 제품 선택을 공유해서, 페이지를 옮겨도 보던 제품이 그대로 남는다
        product = st.segmented_control(
            "제품", products, default=products[0], required=True,
            label_visibility="collapsed", key=f"{KEY_PREFIX}product_switch",
            width="stretch",
        ) or products[0]
    with right:
        # DC HOLD LINK 는 이 페이지에 없다 -- hold 를 푸는 화면이 아니다
        st.markdown(
            f"<div style='text-align:right; font-size:0.8rem; color:#888;'>"
            f"(Latest Data : {data_loaded_at})</div>",
            unsafe_allow_html=True,
        )

    q = (search or "").strip().lower()
    items = [c for c in item_columns(trend_frames[product])
             if not q or q in str(c).lower()]
    if not items:
        st.info(f'"{search}" 와 맞는 item 이 없습니다.' if q
                else "이 제품의 trend 에 item 컬럼이 없습니다.")
        return

    selection = st.session_state.get(sel_key)
    # 선택은 제품을 넘어가면 뜻이 없다 (다른 제품엔 그 wafer 가 없다)
    if selection and selection.get("product") != product:
        selection = None
        st.session_state[sel_key] = None

    seen_key = f"{KEY_PREFIX}wac_seen"
    nonce_key = f"{KEY_PREFIX}wac_nonce"
    seen = st.session_state.setdefault(seen_key, {})
    nonce = st.session_state.setdefault(nonce_key, 0)

    # 선택 문구가 들어갈 자리를 먼저 잡아둔다. 클릭은 아래에서 차트를
    # 그리기 직전에 읽는데, 그 결과를 여기 위에 써야 하기 때문이다.
    msg_slot = st.container()

    # 한 번에 그릴 차트 수. 클릭 한 번에 이만큼을 다시 그려 브라우저로
    # 넘기므로, item 이 많은 제품에서 이걸 키우면 클릭이 그만큼 느려진다.
    per_page = st.session_state.setdefault(f"{KEY_PREFIX}wac_per_page", 10)
    n_pages = max(1, -(-len(items) // per_page))
    page_idx = 0
    if n_pages > 1:
        nav_col, size_col, _sp = st.columns([2, 1.4, 4])
        with nav_col:
            page_idx = st.select_slider(
                "차트 페이지",
                options=list(range(n_pages)),
                format_func=lambda i: f"{i * per_page + 1}~"
                                      f"{min((i + 1) * per_page, len(items))} / {len(items)}",
                key=f"{KEY_PREFIX}wac_page_idx_{product}_{q}_{per_page}",
                label_visibility="collapsed",
            )
        with size_col:
            new_size = st.selectbox(
                "한 번에", [4, 10, 20, 50], index=[4, 10, 20, 50].index(per_page),
                format_func=lambda n: f"{n}개씩", label_visibility="collapsed",
                key=f"{KEY_PREFIX}wac_per_page_pick",
            )
            if new_size != per_page:
                st.session_state[f"{KEY_PREFIX}wac_per_page"] = new_size
                st.rerun()
    shown = items[page_idx * per_page:(page_idx + 1) * per_page]
    chart_keys = {c: f"{KEY_PREFIX}wac_chart_{product}_{c}_{nonce}" for c in shown}

    # 차트를 그리기 '전에' 위젯 상태를 읽는다. st.plotly_chart 가 돌려주는
    # 값은 st.session_state[key] 와 같은 것이라, 그리는 도중에 확인하면
    # 앞쪽 차트는 이미 옛 선택으로 그려진 뒤라서 st.rerun() 으로 한 바퀴를
    # 더 돌아야 한다. 먼저 읽으면 클릭 한 번에 페이지를 두 번 그리던 것이
    # 한 번으로 준다 (차트가 10개면 그림 10장을 덜 만들어 덜 보낸다).
    #
    # '어느 차트가 방금 눌린 것인가' 는 상태만 봐서는 알 수 없다 -- 눌렸던
    # 차트는 그 뒤로도 계속 같은 점을 보고하기 때문이다. 그래서 차트마다
    # 마지막으로 본 값을 seen 에 적어두고, 그것과 달라진 차트만 새 클릭으로
    # 친다.
    for item_col in shown:
        key = chart_keys[item_col]
        cd = clicked_customdata(st.session_state.get(key))
        pair = (cd[0], cd[1]) if cd else None
        if seen.get(key) != pair:
            seen[key] = pair
            if pair is not None:
                selection = {"product": product,
                             "root_lot_id": pair[0], "wafer_id": pair[1]}
                st.session_state[sel_key] = selection

    with msg_slot:
        msg_col, clear_col = st.columns([6, 1])
        with msg_col:
            if selection:
                st.markdown(
                    f"<div style='font-size:0.85rem; color:#d33; font-weight:600;'>"
                    f"선택한 wafer : {norm_lot(selection['root_lot_id'])} "
                    f"#{norm_wafer(selection['wafer_id'])}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("타점을 누르면 그 wafer 가 모든 차트에서 같이 커집니다.")
        with clear_col:
            # 정적 리포트는 같은 타점을 다시 눌러 해제하지만, Streamlit 은
            # 같은 점을 다시 눌러도 위젯 값이 그대로라 아무 일도 일어나지
            # 않는다. 그래서 해제는 버튼으로 둔다. nonce 를 올려 차트 key 를
            # 통째로 바꾸는 이유는, 안 그러면 차트들이 방금 해제한 선택을
            # 계속 들고 있어서 같은 타점을 다시 눌러도 '바뀐 게 없다' 가
            # 되기 때문이다.
            if selection and st.button("선택 해제", key=f"{KEY_PREFIX}wac_clear",
                                       width="stretch"):
                st.session_state[sel_key] = None
                st.session_state[nonce_key] = nonce + 1
                st.session_state[seen_key] = {}
                st.rerun()

    stamp = data_loaded_at
    for row_start in range(0, len(shown), WAC_GRID_COLS):
        cols = st.columns(WAC_GRID_COLS)
        for col, item_col in zip(cols, shown[row_start:row_start + WAC_GRID_COLS]):
            with col, st.container(border=True):
                st.markdown(
                    f"<div style='font-weight:700; font-size:0.95rem;'>{item_col}</div>",
                    unsafe_allow_html=True,
                )
                pts = wac_item_points(product, item_col, stamp)
                if pts["spec_rows"].empty:
                    # 관리선이 없으면 OUT 판정을 할 수 없어 전부 회색이 된다.
                    # 그 사실을 안 적어두면 "이 item 은 다 정상" 으로 읽힌다.
                    st.markdown(
                        f"<div style='font-size:0.75rem; color:#b45309;'>"
                        f"관리선 없음 (spec 에 {item_col} 이 없습니다)</div>",
                        unsafe_allow_html=True,
                    )
                key = chart_keys[item_col]
                event = st.plotly_chart(
                    build_wac_scatter(pts, item_col, selection),
                    width="stretch", on_select="rerun", selection_mode="points",
                    key=key,
                )
                # 보통은 위에서 미리 읽어 이미 반영돼 있다. 여기는 그게
                # 빗나갔을 때를 위한 그물이다 (streamlit 이 위젯 상태를
                # 담는 모양을 바꾸면 위쪽 사전 확인이 조용히 아무것도 못
                # 찾게 되는데, 그러면 클릭이 통째로 안 먹는다).
                cd = clicked_customdata(event)
                if cd:
                    pair = (cd[0], cd[1])
                    if seen.get(key) != pair:
                        seen[key] = pair
                        st.session_state[sel_key] = {
                            "product": product,
                            "root_lot_id": pair[0], "wafer_id": pair[1],
                        }
                        # 이미 그린 차트들은 옛 선택으로 그려져 있다. 여기서
                        # 다시 돌려야 모든 차트에 한꺼번에 반영된다.
                        st.rerun()


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
    frames = load_data()
    data_loaded_at, problems, data_warnings = frames[-3:]
    product_dc, trend_frames, spec_frames, _split_frames = frames_by_product(frames)

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

    # the comment box is a disabled (read-only) text_area, which streamlit
    # renders in light gray by default; override to black and slightly larger
    # so it's actually legible. -webkit-text-fill-color is needed too since
    # some browsers ignore `color` on a disabled field and only honor this.
    # Drawn before the page switch below, since both pages use these rules.
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

    # 최상단 페이지 전환. 정적 리포트와 같은 두 화면이고 기본은 DC OCAP.
    # required=True 는 여기서도 같은 이유다 -- 없으면 눌린 걸 다시 눌러
    # 아무 페이지도 안 골라진 상태가 된다.
    page = st.segmented_control(
        "페이지", PAGES, default=PAGES[0], required=True,
        label_visibility="collapsed", key=f"{KEY_PREFIX}page",
    ) or PAGES[0]
    if page == "WAC Trend":
        render_wac_page(product_dc, trend_frames, spec_frames, data_loaded_at)
        return

    # the counts sit inside the h1 so their 0.42em resolves against the same
    # heading size -- that keeps them consistent without having to hardcode
    # whatever px streamlit's h1 currently renders at. "Latest Data" used to
    # sit here too but now rides just above the trend panel, next to the
    # DC HOLD link.
    new_counts = ", ".join(f"{p} : {count_new_holds(product_dc[p])}건" for p in product_dc)
    st.markdown(
        f"# Hold 현황 "
        f"<span style='display:inline-block; vertical-align:bottom; {HEAD_META_STYLE}'>"
        f"신규 hold 건수<br>{new_counts}</span>",
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
        empty_note = None
        if full_dc_df is None or full_dc_df.empty or "hold_time" not in full_dc_df.columns:
            full_dc_df = pd.DataFrame(columns=DC_REQUIRED)
            empty_note = f"{selected_product} 은(는) 조회 기간에 hold 건이 없습니다."
        # a clicked chart point is resolved against the full (unfiltered) set,
        # since its history shouldn't disappear just because the current status
        # filter happens to hide the lot it belongs to
        dc_df = filter_by_status(full_dc_df, status_filter)
        grouped = group_holds(dc_df)
        # 빈 표만 남으면 고장난 화면처럼 보인다. 비어 있는 이유를 그 자리에서
        # 말해준다 (정적 리포트의 리스트도 같은 문구를 쓴다)
        if empty_note is None and grouped.empty:
            empty_note = (
                "조치 대기 중인 hold 가 없습니다. '전체' 나 '이력' 을 눌러보세요."
                if status_filter == "hold" else "이 조건에 맞는 행이 없습니다."
            )
        if empty_note:
            st.caption(empty_note)

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

                    if tdf[item_col].notna().sum() == 0:
                        st.warning(f"{item_id} 은(는) 이 제품 trend 에 측정값이 없습니다.")

                    fig = build_scatter(
                        tdf, item_col, bad_pairs, str(lot_id), legend_field,
                        spec_frames.get(product),
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


def _spec_for_export(spec_df) -> pd.DataFrame:
    """spec 을 브라우저로 보낼 모양으로: 필요한 컬럼만, from_time 은 datetime."""
    if not isinstance(spec_df, pd.DataFrame) or spec_df.empty:
        return pd.DataFrame(columns=SPEC_REQUIRED)
    out = spec_df[SPEC_REQUIRED].copy()
    out["from_time"] = pd.to_datetime(out["from_time"], errors="coerce")
    # 관리선까지 넣어 정렬하는 이유는 item_spec_rows() 쪽과 같다. 브라우저의
    # sort 는 안정 정렬이라, 여기 순서가 그대로 남아 같은 시각에 두 벌이
    # 있어도 파이썬 화면과 정적 리포트가 같은 쪽을 고른다.
    return (out.dropna(subset=["from_time"])
            .sort_values(["item_id", "from_time", *LIMIT_COLS], na_position="first"))


def reachable_lots(dc_df, trend_df) -> set:
    """EINECN 버튼이 열릴 수 있는 root_lot_id 전부.

    버튼은 리스트에서 고른 hold 의 lot, 아니면 차트에서 누른 타점의 lot 으로
    열린다. 차트에는 그 item 의 trend 타점이 전부(회색 배경까지) 찍히고 아무
    거나 누를 수 있으므로, dc 의 lot 뿐 아니라 trend 의 lot 도 다 열릴 수 있다.
    """
    lots = set()
    for df in (dc_df, trend_df):
        if isinstance(df, pd.DataFrame) and "root_lot_id" in df.columns:
            lots |= {norm_lot(v) for v in df["root_lot_id"]}
    return lots


def _split_for_export(split_df, keep_lots=None) -> pd.DataFrame:
    """split 을 EINECN 팝업이 쓸 모양으로: 필요한 칸만, 정해진 순서로.

    einecn_no 하나가 step 여러 개에 걸린다 -- 하나의 test 를 step 여럿에
    묶어서 돌리기 때문이다. 그러니 einecn_no 로 묶어 한 줄로 접으면 안 되고,
    step 마다 한 줄로 둔다 (step 이 다르면 적용된 wafer 도 다르다). 대신
    einecn_no 로 먼저 정렬해서 같은 test 의 step 들이 붙어 나오게 한다.

    keep_lots 를 주면 그 lot 의 행만 남긴다. split 은 라인 전체 이력이라
    dc/trend 조회 기간 밖의 lot 이 잔뜩 들어 있는데, 그 lot 은 화면에서
    고를 수가 없으니 팝업도 열릴 수가 없다. 실을 이유가 없는 데이터다
    (실측: 그런 행이 65% 면 payload 1.82MB -> 0.64MB).
    """
    if not isinstance(split_df, pd.DataFrame) or split_df.empty:
        return pd.DataFrame(columns=SPLIT_REQUIRED)
    missing = [c for c in SPLIT_REQUIRED if c not in split_df.columns]
    if missing:
        raise SystemExit(
            "split 에 EINECN 팝업이 요구하는 컬럼이 없습니다: "
            + ", ".join(map(str, missing))
            + f"\n실제 컬럼: {list(split_df.columns)}"
            + "\n1~25 는 wafer 번호 칸입니다 (comp_id_list 를 펼친 결과)."
        )
    out = split_df[SPLIT_REQUIRED]
    if keep_lots is not None:
        # 화면 쪽 lot 비교와 같은 규칙으로 거른다. 여기만 원본 문자열로
        # 비교하면, 공백 하나 차이로 멀쩡한 이력이 통째로 빠진다.
        out = out[out["root_lot_id"].map(norm_lot).isin(keep_lots)]
    # 팝업은 test(einecn_no) 단위로 칸을 병합해 읽으므로 같은 test 의 줄이
    # 흩어지면 안 된다. 그러면서 step 순서로도 읽혀야 하니, test 안에서는
    # step_seq 순으로 늘어놓고, test 끼리는 그 test 의 첫 step_seq 순으로
    # 놓는다 -- 이름순으로 놓으면 step_seq 칸이 위아래로 튄다.
    # 정렬은 결정적이어야 한다: 시간마다 다시 만드는 파일이라, 순서가
    # 흔들리면 내용이 같아도 매번 다른 파일이 올라간다.
    out = out.sort_values(["root_lot_id", "einecn_no", "step_seq"], kind="stable")
    first_step = out.groupby(["root_lot_id", "einecn_no"], sort=False)["step_seq"].transform("min")
    return (out.assign(_first_step=first_step)
            .sort_values(["root_lot_id", "_first_step", "einecn_no", "step_seq"], kind="stable")
            .drop(columns="_first_step"))


def build_dc_ocap_html() -> Path:
    """Generate dc_ocap.html from the current pull_data() and return its path."""
    # 제품 순서까지 한 곳에서 정한다. 여기서 순서가 어긋나면 리포트 머리글의
    # 건수만 제품 전환 버튼과 다른 순서로 나온다 (예전에 그랬다).
    product_dc, product_trend, product_spec, product_split = frames_by_product(pull_data())

    # same validation the Streamlit page runs before trusting the data --
    # a bad schema should fail the scheduled build loudly rather than ship
    # a broken dc_ocap.html. Warnings are printed but must not stop the
    # build: this runs hourly and uploads to S3, so failing over a few
    # wafers missing from trend would freeze the portal on a stale report.
    problems, warnings = check_data(product_dc, product_trend, product_spec)
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
            p: item_columns(product_trend[p])
            for p in product_trend
        },
        # 관리선 개정 이력. from_time 은 반드시 datetime 으로 맞춰서 넘긴다:
        # 브라우저는 tkout_time 과 문자열로 비교하는데, tkout_time 은 항상
        # isoformat("...T...") 이라 from_time 이 "2026-08-01 01:00:00" 처럼
        # 공백 구분 문자열로 오면 공백(0x20) < "T"(0x54) 때문에 규격이 바뀐
        # 당일 측정에 이전 규격이 적용된다. 여기서 한 번 변환해 두면 어떤
        # 형식으로 들어와도 양쪽이 같은 표기가 된다.
        "spec": {p: _columns(_spec_for_export(product_spec[p])) for p in product_spec},
        # EIN/ECN 적용 이력. 차트 밑 EINECN 버튼이 (제품, root_lot_id) 로
        # 찾아 팝업에 띄운다. 화면에서 열릴 수 없는 lot 은 싣지 않는다.
        "split": {
            p: _columns(_split_for_export(
                product_split[p],
                reachable_lots(product_dc.get(p), product_trend.get(p))))
            for p in product_split
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

    # 어느 데이터가 파일을 키우는지 매번 찍어 둔다. 이 파일은 한 시간마다
    # 다시 만들어 올라가는데, 조회 조건 하나 넓히면 조용히 몇 배가 될 수
    # 있다 -- 열어 보고 나서야 아는 것보다 여기서 보이는 편이 낫다.
    for name in ("dc", "trend", "spec", "split"):
        chunk = json.dumps(data[name], ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
        rows = sum(len(next(iter(c.values()), [])) for c in data[name].values())
        print(f"  {name:6s} {rows:>9,}행  json {len(chunk)/1024/1024:6.2f} MB"
              f"  -> {len(chunk)/len(payload)*100:4.1f}% of payload")

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

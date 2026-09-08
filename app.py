"""DC OCAP 정적 리포트를 만든다 (dc_ocap.html).

사내 시스템에서 데이터를 뽑아 dc_ocap_template.html 에 밀어넣고, 파이썬이
없어도 열리는 파일 하나로 내보낸다. 스케줄러가 한 시간마다 이 파일을
실행하고, 나온 dc_ocap.html 을 포털이 보는 자리(S3)에 올린다.

    python app.py          -> dc_ocap.html 생성
    python diagnose.py     -> 데이터가 왜 안 보이는지 단계별 진단

화면 동작(목록, 차트, 통계, EINECN 팝업)은 전부 dc_ocap_template.html 안의
자바스크립트에 있다. 여기 파이썬이 하는 일은 '무엇을 넘길지' 까지다.

======================================================================
DATA PREP (mock — 사내 조회를 대신하는 가짜 데이터. 실제 쿼리는 여기 올릴
수 없어서 형태만 같게 만들어 둔 것이다.)

이 구역을 통째로 사내 조회로 갈아끼우면 된다. 조건은 하나, pull_data() 가
아래 열두 개를 이 순서로 돌려주는 것:

    uly_dc    / sol_dc    / tts_dc
    uly_trend / sol_trend / tts_trend
    uly_spec  / sol_spec  / tts_spec
    uly_split / sol_split / tts_split

순서가 곧 계약이다 -- 이름이 아니라 위치로 읽는다. 순서를 틀리면 오류 없이
다른 제품 데이터가 다른 제품 이름표를 달고 나온다.

조회는 반드시 pull_data() 안에 둔다. 모듈 수준에 두면 이 파일을 import
하는 것만으로 사내 시스템을 때리게 된다.

이 구역은 아래쪽에 기대지 않는다 (자기 import 만 쓴다). 그래서 통째로
갈아끼워도 리포트 쪽이 깨지지 않는다.
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
# wafer 칸(1~25) 목록은 아래 SPLIT_WAFER_COLUMNS 하나뿐이다. 여기서 또
# 만들면 두 벌이 되고, 이름까지 비슷해서(SPLIT_WAFER_COLS vs _COLUMNS) 어느
# 쪽을 고쳤는지 알기 어려워진다. 함수 안에서 쓰므로 정의 순서는 상관없다.

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
                    for col in SPLIT_WAFER_COLUMNS:
                        row[col] = "V" if int(col) in hit else ""
                    rows.append(row)

    return pd.DataFrame(rows)[
        ["einecn_no", "root_lot_id", "ppid", "ein_ecn_type"]
        + SPLIT_WAFER_COLUMNS
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

    사내 조회는 반드시 이 함수 안에 둔다. 모듈 수준에 두면 이 파일을
    import 하는 것만으로 사내 시스템을 때린다.
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
# 여기부터 report (위 구역을 갈아끼워도 이 아래는 그대로 쓴다)
# ======================================================================

# imported here rather than at the top of the file so this section keeps
# working if the data prep above is replaced wholesale
import base64
import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.offline as pyo

# KST is pinned at UTC+9 rather than read from the host clock, so the
# header timestamp stays correct wherever the app is deployed.
KST = timezone(timedelta(hours=9))

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


# ======================================================================
# 정적 리포트(dc_ocap_template.html)와 반드시 같아야 하는 값들.
#
# 여기가 유일한 출처다. build_dc_ocap_html() 이 shared_constants_js() 로
# 브라우저에 넘겨주므로, 템플릿에는 같은 숫자를 다시 적지 않는다.
# 예전에는 양쪽에 손으로 적어 두었다가 조용히 갈라졌다 -- 배경 타점 상한이
# 한쪽은 2500, 다른 쪽은 1200 이 되어 같은 데이터로 다른 그림을 그렸다.
# ======================================================================
LEGEND_FIELD_OPTIONS = {
    "없음 (기본)": None,
    "probe_card_id": "probe_card_id",
    "eqp_id": "eqp_id",
    "lot_type": "lot_type",
    "rw_cnt": "rw_cnt",
}
# red means "past the scrap limit" and blue "past the control limit"
LIMIT_COLORS = {"scrap": "red", "control": "blue"}
WAC_GRAY, WAC_SAME = "lightgray", "dimgray"
WAC_SIZE, WAC_SIZE_SEL = 6, 14
# 배경(회색) 타점이 이보다 많으면 고르게 솎는다. 260px 차트에 수천 점을
# 찍어봐야 대부분 겹치는데, plotly 는 타점마다 <path> 를 하나씩 만들기
# 때문에 비용은 점 수에 그대로 비례한다 (실측: 2500 점이면 한 번 훑을 때
# 긴 작업 합계 7.9초, 1200 점이면 5.0초).
# CL OUT / SL OUT 은 절대 솎지 않는다 -- 그게 봐야 할 신호다.
WAC_MAX_GRAY = 1200

# 이상값(어이없이 큰 값) 판정. median + MAD 를 쓰는 이유는 평균/표준편차가
# 센티널 값 하나에 통째로 망가지기 때문이다.
#
# 5로 잡은 이유: 관리선(UCL/LCL)이 보통 ±3시그마다. 3으로 자르면 걸러야 할
# 이상값이 아니라 정작 봐야 할 CL OUT 타점을 지우게 된다 (모의 데이터
# 8000점에서 3시그마는 20개를 걸렀는데 그 중 18개가 멀쩡한 값이었고,
# 5시그마는 진짜 이상값 2개만 걸렀다).
OUTLIER_K = 5
OUTLIER_MIN_N = 20      # 이보다 적으면 흩어진 정도를 못 믿는다


def shared_constants_js() -> str:
    """위 값들을 브라우저가 읽을 수 있는 한 줄짜리 JS 로 만든다."""
    return "const SHARED = " + json.dumps({
        "legendFieldOptions": LEGEND_FIELD_OPTIONS,
        "limitColors": LIMIT_COLORS,
        "wacGray": WAC_GRAY,
        "wacSame": WAC_SAME,
        "wacSize": WAC_SIZE,
        "wacSizeSel": WAC_SIZE_SEL,
        "wacMaxGray": WAC_MAX_GRAY,
        "outlierK": OUTLIER_K,
        "outlierMinN": OUTLIER_MIN_N,
        "splitWaferColumns": SPLIT_WAFER_COLUMNS,
        "limitCols": list(LIMIT_COLS),
    }, ensure_ascii=False, separators=(",", ":")) + ";"






def item_columns(trend_df) -> list:
    """The measurement columns of a trend frame (everything but its metadata)."""
    return [c for c in trend_df.columns if str(c) not in META_TREND_COLS]






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
               spec_frames: dict | None = None,
               split_frames: dict | None = None) -> tuple[list[str], list[str]]:
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

        # split(EIN/ECN 적용 이력). 화면에서는 EINECN 버튼을 눌러야 보이는
        # 것이라, 잘못돼도 나머지 대시보드는 멀쩡히 돈다 -- 그래서 전부
        # 경고다. 다만 정적 리포트를 만들 때는 필요한 칸이 없으면 빌드가
        # 멈추므로, 그 전에 여기서 같은 목록으로 먼저 알려준다.
        split_df = (split_frames or {}).get(product)
        if split_frames is None:
            pass                                   # 부르는 쪽이 split 을 안 넘겼다
        elif not isinstance(split_df, pd.DataFrame):
            warnings.append(
                f"{product.lower()}_split: DataFrame 이 아닙니다 "
                f"({type(split_df).__name__}). EINECN 팝업이 비어 보입니다."
            )
        elif split_df.empty:
            warnings.append(
                f"{product.lower()}_split: 비어 있습니다. "
                f"split 을 process_id 로 자르는 조건을 확인하세요."
            )
        else:
            missing = [c for c in SPLIT_REQUIRED if c not in split_df.columns]
            if missing:
                warnings.append(
                    f"{product.lower()}_split: 컬럼 없음 -> {', '.join(map(str, missing[:8]))}"
                    + (" ..." if len(missing) > 8 else "")
                    + ". 1~25 는 wafer 번호 칸입니다 (comp_id_list 를 펼친 결과). "
                    "정적 리포트 빌드는 이 상태로는 멈춥니다."
                )
            else:
                # 그 lot 의 이력을 찾는 열쇠라, 여기가 어긋나면 팝업이 늘
                # 비어 있는데 화면에는 아무 오류도 안 뜬다
                trend_lots = (set(trend_df["root_lot_id"].map(norm_lot))
                              if isinstance(trend_df, pd.DataFrame)
                              and "root_lot_id" in trend_df.columns else set())
                split_lots = set(split_df["root_lot_id"].map(norm_lot))
                if trend_lots and not (split_lots & trend_lots):
                    warnings.append(
                        f"{product.lower()}_split: root_lot_id 가 "
                        f"{product.lower()}_trend 와 하나도 겹치지 않습니다 "
                        f"(EINECN 팝업이 늘 비어 보입니다). split 예시 "
                        f"{sorted(split_lots)[:3]}"
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












# ======================================================================
# dc_ocap.html 만들기.
#
# 사내 시스템에 닿는 곳에서 이 파일을 주기적으로 실행하고, 나온 파일을
# 포털이 보는 자리(S3)에 올린다. 여기서는 로컬 파일만 쓴다 -- 올리는 데
# 필요한 자격증명은 이 저장소에 없다.
#
# 화면 동작은 전부 dc_ocap_template.html 안의 자바스크립트에 있다. 파이썬은
# 데이터를 골라 넣어줄 뿐이고, 그래서 두 쪽이 같은 판정을 내려야 하는 값
# (솎는 기준, 이상값 시그마, 색)은 위 SHARED 한 곳에서만 나온다.
# ======================================================================

# __file__ only exists when this runs as an actual .py script (which is
# how the scheduler will call it); it's undefined in a notebook cell, so
# fall back to the current working directory there
HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
TEMPLATE_PATH = HERE / "dc_ocap_template.html"
OUTPUT_PATH = HERE / "dc_ocap.html"


class BuildError(Exception):
    """리포트를 만들 수 없다. 메시지는 그대로 사람이 읽는 안내문이다.

    SystemExit 대신 쓴다: 예전에는 SystemExit 를 던졌는데, 그건 이 함수를
    import 해서 쓰는 쪽(스케줄러 스크립트, 노트북, 테스트)의 프로세스를
    통째로 죽여 버린다. 프로세스를 끝낼지는 부르는 쪽이 정할 일이다 --
    이 파일을 직접 실행했을 때만 __main__ 에서 종료 코드로 바꾼다.
    """



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
        raise BuildError(
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

    # 스키마가 틀리면 깨진 dc_ocap.html 을 내보내지 말고 시끄럽게 멈춘다. Warnings are printed but must not stop the
    # build: this runs hourly and uploads to S3, so failing over a few
    # wafers missing from trend would freeze the portal on a stale report.
    problems, warnings = check_data(product_dc, product_trend, product_spec, product_split)
    if problems:
        raise BuildError(
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
            raise BuildError(
                f"{TEMPLATE_PATH.name} 에서 '{placeholder}' 를 찾지 못했습니다.\n"
                f"app.py 와 {TEMPLATE_PATH.name} 의 버전이 서로 다른 것 같습니다 "
                f"(둘은 같은 커밋의 짝으로 써야 합니다).\n"
                f"두 파일을 함께 최신으로 받아서 다시 실행하세요."
            )
        return text.replace(placeholder, value)

    html = fill(template, "__GENERATED_AT__", generated_at)
    html = fill(html, "__DATA_B64__", encoded)
    # 두 화면이 같은 판정을 내려야 하는 값들. 템플릿에 같은 숫자를 다시
    # 적어두지 않으므로, 여기서 안 넣으면 페이지가 아예 안 뜬다 (fill 이
    # 플레이스홀더가 없으면 멈추므로 조용히 빠질 수는 없다)
    html = fill(html, "/*__SHARED_CONSTANTS__*/", shared_constants_js())
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
# 이 파일을 직접 실행하면 리포트를 만든다. import 하면 아무 일도 안 한다
# -- 스케줄러 스크립트나 노트북에서 build_dc_ocap_html() 만 따로 부를 수
# 있게 하기 위해서다.
if __name__ == "__main__":
    # BuildError 는 사람이 읽는 안내문이다. traceback 없이 그대로 보여
    # 주고 0 이 아닌 코드로 끝낸다 -- 스케줄러가 실패를 알아채야 한다.
    try:
        build_dc_ocap_html()
    except BuildError as err:
        print(f"\n빌드 실패: {err}", file=sys.stderr)
        raise SystemExit(1)

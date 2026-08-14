"""
Mock data generator for the OCAP dashboard.

`dc` mimics the hold-event dataframe pulled from the datalake: when a
product/lot gets put on HOLD due to a measurement value being out of
spec/control, one row is recorded here.

`uly` / `tts` / `sol` mimic the per-product wide-format probe test
("trend") dataframes: one row per wafer measurement, item1..itemN
holding the measured values.

`dc` rows are sampled from the trend dataframes (same root_lot_id /
wafer_id / item column) so the dashboard can actually look up a hold
event's trend history.

Real data source and pull logic are internal/confidential; this module
only produces structurally-similar synthetic data for local dev of the
Streamlit dashboard.
"""

import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

LOT_ID_CHARS = list(string.ascii_uppercase + string.digits)

LINE_IDS = ["M14", "M16", "L1", "L2"]
PROCESS_IDS = ["PHOTO", "ETCH", "DEP", "CMP", "DIFF", "IMP", "CLEAN"]
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

# per-product item value profile: (center, spread) so each product's
# item columns look distinct from one another
PRODUCT_CONFIG = {
    "ULY": {"n_items": 18, "center": 50, "spread": 6, "seed": 101},
    "TTS": {"n_items": 22, "center": 120, "spread": 10, "seed": 102},
    "SOL": {"n_items": 15, "center": 8, "spread": 1.5, "seed": 103},
}


def _random_datetime(rng: np.random.Generator, start: datetime, end: datetime, size: int) -> pd.Series:
    delta_seconds = int((end - start).total_seconds())
    offsets = rng.integers(0, delta_seconds, size=size)
    return pd.Series([start + timedelta(seconds=int(s)) for s in offsets])


def generate_probe_df(product: str, n_rows: int = 300) -> pd.DataFrame:
    """Generate a mock wide-format probe test dataframe for a single product.

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
    for lot, wafer_no in base_wafers:
        wafer_id = f"{lot}.{wafer_no:02d}"
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


# convenience instances matching the real-world variable names (uly/tts/sol)
uly = generate_probe_df("ULY")
tts = generate_probe_df("TTS")
sol = generate_probe_df("SOL")

TREND_FRAMES = {"ULY": uly, "TTS": tts, "SOL": sol}


def generate_dc(trend_frames: dict = TREND_FRAMES, n_rows: int = 150, seed: int | None = 42) -> pd.DataFrame:
    """Generate a mock `dc` (hold event) dataframe with n_rows rows.

    Each hold event is tied to a real (root_lot_id, wafer_id, item column)
    combination sampled from `trend_frames`, so it can be looked up in the
    corresponding trend dataframe on the dashboard.
    """
    rng = np.random.default_rng(seed)
    products = list(trend_frames.keys())

    rows = []
    for _ in range(n_rows):
        product = rng.choice(products)
        tdf = trend_frames[product]
        base_rows = tdf[tdf["rw_cnt"] == 0]
        src = base_rows.iloc[rng.integers(0, len(base_rows))]
        item_cols = [c for c in tdf.columns if c.startswith("item")]
        item_id = rng.choice(item_cols)

        cfg = PRODUCT_CONFIG[product]
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
                "process_id": rng.choice(PROCESS_IDS),
                "sub_item_id": f"{item_id}_{rng.integers(1, 4)}",
            }
        )

    dc = pd.DataFrame(rows)
    dc["hold_time"] = _random_datetime(
        rng, datetime(2026, 7, 1), datetime(2026, 8, 14, 23, 59, 59), n_rows
    ).sort_values().reset_index(drop=True)

    return dc[
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


dc = generate_dc()


if __name__ == "__main__":
    print(dc.head(10).to_string(index=False))
    print("\ndc shape:", dc.shape)
    print("\ndtypes:\n", dc.dtypes)

    print("\n" + "=" * 80)
    for name, df in TREND_FRAMES.items():
        print(f"\n{name.lower()} shape:", df.shape)
        print(df.head(3).to_string(index=False))

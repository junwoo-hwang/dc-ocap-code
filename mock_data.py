"""
Mock data generator for the OCAP dashboard.

`dc` mimics the hold-event dataframe pulled from the datalake:
when a product/lot gets put on HOLD due to a measurement value
being out of spec/control, one row is recorded here.

Real data source and pull logic are internal/confidential; this module
only produces structurally-similar synthetic data for local dev of the
Streamlit dashboard.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

LINE_IDS = ["M14", "M16", "L1", "L2"]
PROCESS_IDS = ["PHOTO", "ETCH", "DEP", "CMP", "DIFF", "IMP", "CLEAN"]
ITEM_IDS = ["THICKNESS", "CD", "OVERLAY", "RESISTANCE", "PARTICLE", "STRESS", "PROFILE"]
HOLD_REASONS = [
    "OOC (Out of Control)",
    "SPEC OUT (USL Exceed)",
    "SPEC OUT (LSL Exceed)",
    "EQP ALARM",
    "TREND WARNING (7 POINT RUN)",
    "SUDDEN SHIFT",
    "MEASUREMENT DELAY",
]


def _random_datetime(start: datetime, end: datetime, size: int) -> pd.Series:
    delta_seconds = int((end - start).total_seconds())
    offsets = RNG.integers(0, delta_seconds, size=size)
    return pd.Series([start + timedelta(seconds=int(s)) for s in offsets])


def generate_dc(n_rows: int = 300, seed: int | None = 42) -> pd.DataFrame:
    """Generate a mock `dc` (hold event) dataframe with n_rows rows."""
    rng = np.random.default_rng(seed)

    n_lots = max(1, n_rows // 5)
    root_lot_ids = [f"P{rng.integers(1, 9)}L{rng.integers(1000, 9999)}.{rng.integers(0,999):03d}" for _ in range(n_lots)]

    line_id = rng.choice(LINE_IDS, size=n_rows)
    process_id = rng.choice(PROCESS_IDS, size=n_rows)
    item_id = rng.choice(ITEM_IDS, size=n_rows)
    sub_item_id = [f"{it}_{rng.integers(1,4)}" for it in item_id]
    hold_inform = rng.choice(HOLD_REASONS, size=n_rows)

    root_lot_id = rng.choice(root_lot_ids, size=n_rows)
    wafer_no = rng.integers(1, 26, size=n_rows)  # 25 wafers per lot typical
    wafer_id = [f"{lot}.{w:02d}" for lot, w in zip(root_lot_id, wafer_no)]

    step_seq = rng.integers(10, 500, size=n_rows)

    # spec/control limits: usl/lsl wider than ucl/lcl, centered around a base value
    base = rng.normal(loc=100, scale=15, size=n_rows)
    usl = base + rng.uniform(8, 15, size=n_rows)
    lsl = base - rng.uniform(8, 15, size=n_rows)
    ucl = base + rng.uniform(3, 7, size=n_rows)
    lcl = base - rng.uniform(3, 7, size=n_rows)

    hold_time = _random_datetime(
        datetime(2026, 7, 1), datetime(2026, 8, 12, 23, 59, 59), n_rows
    ).sort_values().reset_index(drop=True)

    dc = pd.DataFrame(
        {
            "root_lot_id": root_lot_id,
            "wafer_id": wafer_id,
            "hold_time": hold_time,
            "item_id": item_id,
            "hold_inform": hold_inform,
            "ucl": ucl.round(3),
            "lcl": lcl.round(3),
            "usl": usl.round(3),
            "lsl": lsl.round(3),
            "step_seq": step_seq,
            "line_id": line_id,
            "process_id": process_id,
            "sub_item_id": sub_item_id,
        }
    )

    return dc


if __name__ == "__main__":
    dc = generate_dc()
    print(dc.head(10).to_string(index=False))
    print("\nshape:", dc.shape)
    print("\ndtypes:\n", dc.dtypes)

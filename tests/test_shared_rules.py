"""파이썬 화면과 정적 리포트가 같은 판정을 내리는지.

이 파일이 있는 이유: 예전에는 같은 규칙을 app.py 와 dc_ocap_template.html
양쪽에 손으로 적어 두었고, 그러다 조용히 갈라졌다.

  - 배경 타점을 솎는 상한이 한쪽은 2500, 다른 쪽은 1200 이었다
  - 5시그마 이상값 제거가 브라우저 쪽에만 있었다
  - '정확히 0 은 계측 실패' 규칙이 파이썬 쪽에만 남아 있었다 (0 근처인
    item 에서 멀쩡한 값을 지운다는 이유로 브라우저 쪽에서는 지운 규칙)

셋 다 화면에 오류 하나 없이 서로 다른 그림을 그렸다.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app

TEMPLATE = Path(app.TEMPLATE_PATH).read_text(encoding="utf-8")

# 템플릿이 자기 값을 따로 들고 있으면 안 되는 상수들.
# (JS 이름, SHARED 의 키)
SHARED_IN_TEMPLATE = [
    ("LEGEND_FIELD_OPTIONS", "legendFieldOptions"),
    ("LIMIT_COLORS", "limitColors"),
    ("WAC_MAX_GRAY", "wacMaxGray"),
    ("OUTLIER_K", "outlierK"),
    ("OUTLIER_MIN_N", "outlierMinN"),
    ("EIN_WAFERS", "splitWaferColumns"),
    ("LIMIT_COLS", "limitCols"),
]


def test_no_constant_is_written_out_in_both_files():
    """양쪽에 같은 이름으로 값이 따로 적힌 상수가 하나도 없어야 한다.

    이 검사가 있는 이유는 위 목록이 손으로 관리되기 때문이다 -- 나중에
    새 상수를 양쪽에 적으면, 목록에 넣는 걸 잊어도 여기서 걸린다.
    """
    src = Path(app.__file__).read_text(encoding="utf-8")
    py_const = dict(re.findall(r"^([A-Z][A-Z0-9_]{2,})\s*=\s*(.+)$", src, re.M))
    js_const = dict(re.findall(r"^const ([A-Z][A-Z0-9_]{2,})\s*=\s*(.+?);\s*$",
                               TEMPLATE, re.M))
    both = sorted(set(py_const) & set(js_const))
    duplicated = [k for k in both if "SHARED." not in js_const[k]]
    assert not duplicated, (
        f"양쪽에 값이 따로 적힌 상수: {duplicated}. "
        f"app.py 의 SHARED 로 옮기고 템플릿은 거기서 읽게 하세요."
    )


@pytest.mark.parametrize("js_name,shared_key", SHARED_IN_TEMPLATE)
def test_template_reads_shared_constants_instead_of_repeating_them(js_name, shared_key):
    """템플릿은 값을 적어두지 않고 SHARED 에서 읽어야 한다."""
    found = re.findall(rf"^const {js_name}\s*=\s*(.+?);\s*$", TEMPLATE, re.M)
    assert found, f"{js_name} 정의를 템플릿에서 못 찾았습니다"
    assert len(found) == 1, f"{js_name} 정의가 {len(found)}개 있습니다"
    assert f"SHARED.{shared_key}" in found[0], (
        f"{js_name} 이 SHARED.{shared_key} 대신 값을 직접 들고 있습니다: {found[0]}"
    )


def test_shared_constants_js_is_valid_and_complete():
    js = app.shared_constants_js()
    assert js.startswith("const SHARED = ") and js.endswith(";")
    payload = json.loads(js[len("const SHARED = "):-1])
    for _js_name, key in SHARED_IN_TEMPLATE:
        assert key in payload, f"SHARED 에 {key} 가 없습니다"
    assert payload["wacMaxGray"] == app.WAC_MAX_GRAY
    assert payload["outlierK"] == app.OUTLIER_K
    assert payload["splitWaferColumns"] == app.SPLIT_WAFER_COLUMNS


def test_wafer_columns_defined_once():
    """wafer 1~25 목록이 파이썬에 두 벌 있으면 안 된다."""
    src = Path(app.__file__).read_text(encoding="utf-8")
    defs = re.findall(r"^(SPLIT_WAFER[A-Z_]*)\s*=", src, re.M)
    assert defs == ["SPLIT_WAFER_COLUMNS"], f"wafer 칸 정의가 여럿입니다: {defs}"
    assert app.SPLIT_WAFER_COLUMNS == [str(n) for n in range(1, 26)]


# ---------------------------------------------------------------- 이상값

def test_robust_bounds_needs_enough_points():
    assert app.robust_bounds([1.0] * (app.OUTLIER_MIN_N - 1)) is None


def test_robust_bounds_survives_a_sentinel():
    """평균/표준편차였다면 센티널 하나에 통째로 망가진다."""
    values = list(np.linspace(49, 51, 200)) + [9.9e9]
    lo, hi = app.robust_bounds(values)
    assert 40 < lo < 50 < hi < 60, (lo, hi)


def test_robust_bounds_falls_back_to_iqr_when_mad_is_zero():
    values = [5.0] * 150 + list(np.linspace(5.1, 6.0, 60))
    assert app.robust_bounds(values) is not None


def test_is_absurd_flags_only_the_sentinel():
    values = pd.Series(list(np.linspace(49, 51, 200)) + [9.9e9])
    bounds = app.robust_bounds(values)
    absurd = app.is_absurd(values, bounds, None, pd.Series(False, index=values.index))
    assert absurd.sum() == 1 and bool(absurd.iloc[-1])


def test_is_absurd_never_hides_a_held_wafer():
    """hold 사유가 바로 그 극단값인 경우가 많다 -- 숨기면 이유가 사라진다."""
    values = pd.Series(list(np.linspace(49, 51, 200)) + [9.9e9])
    held = pd.Series([False] * 200 + [True])
    bounds = app.robust_bounds(values[~held])
    assert app.is_absurd(values, bounds, None, held).sum() == 0


def test_is_absurd_never_hides_a_value_inside_spec():
    values = pd.Series(list(np.linspace(49.9, 50.1, 200)) + [70.0])
    bounds = app.robust_bounds(values)
    lim = {"lsl": pd.Series(0.0, index=values.index),
           "usl": pd.Series(100.0, index=values.index)}
    assert app.is_absurd(values, bounds, lim, pd.Series(False, index=values.index)).sum() == 0


def test_exact_zeros_survive_on_a_zero_centred_item():
    """예전에 파이썬 쪽에만 남아 있던 '정확히 0 = 계측 실패' 규칙 회귀.

    누설처럼 값이 0 근처인 item 에서는 진짜 0 이 흔하다. 그걸 지우면
    브라우저 화면에는 찍히는 타점이 파이썬 화면에서만 사라진다.
    """
    values = pd.Series([0.0] * 12 + list(np.linspace(-0.5, 0.5, 188)))
    bounds = app.robust_bounds(values)
    absurd = app.is_absurd(values, bounds, None, pd.Series(False, index=values.index))
    assert absurd.sum() == 0, "0 이 이상값으로 잡혔습니다"


def test_python_dc_chart_drops_the_sentinel_but_keeps_the_held_one():
    """build_scatter 가 실제로 이상값을 빼는가 (배경에서만)."""
    trend = app.generate_probe_df("ULY")
    item = app.item_columns(trend)[0]
    trend = trend.reset_index(drop=True)
    trend.loc[0, item] = 9.9e9          # hold 아님 -> 빠져야 한다
    trend.loc[1, item] = 9.9e9          # hold 임   -> 남아야 한다
    held = {(app.norm_lot(trend.loc[1, "root_lot_id"]),
             app.norm_wafer(trend.loc[1, "wafer_id"]))}
    spec = app.generate_spec_for_product("ULY", trend)

    fig = app.build_scatter(trend, item, held, "LOT.1", None, spec, 300)
    drawn = np.concatenate([np.asarray(t.y, dtype="float64") for t in fig.data
                            if getattr(t, "y", None) is not None and len(t.y)])
    assert (drawn > 1e8).sum() == 1, "hold 건 하나만 남아야 합니다"

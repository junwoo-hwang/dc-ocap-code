"""app.py 와 dc_ocap_template.html 이 같은 값을 두 벌 들고 있지 않은지.

화면 동작은 전부 템플릿(자바스크립트)에 있고, 파이썬은 무엇을 넘길지만
정한다. 그런데 '솎는 기준', '이상값 시그마' 처럼 두 파일이 같이 알아야
하는 값이 몇 개 있다. 이걸 양쪽에 손으로 적어 두었더니 실제로 갈라졌다 --
배경 타점 상한이 한쪽은 2500, 다른 쪽은 1200 이 되어, 같은 데이터로 오류
하나 없이 다른 그림을 그렸다.

그래서 지금은 app.py 가 유일한 출처이고 빌드할 때 SHARED 로 넘긴다.
이 파일은 그 규칙이 지켜지는지 본다. 값이 실제로 브라우저에서 먹히는지는
test_report_page.py 가 확인한다.
"""
import json
import re
from pathlib import Path

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

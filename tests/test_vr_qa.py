"""VR_QA 로 시작하는 item 의 음수는 잘못 찍힌 값이다. 절대값으로 바꾼다.

pull_data() 가 준 trend 를 제품별로 나누는 자리(frames_by_product)에서 한 번
바꾼다. 리포트를 만드는 쪽과 진단 스크립트가 다 그 길로 오므로 차트, OUT
판정, 통계가 같은 값을 본다.
"""
from decimal import Decimal

import numpy as np
import pandas as pd

import app


def _trend(**items):
    n = len(next(iter(items.values())))
    base = {"root_lot_id": ["L1"] * n, "wafer_id": list(range(1, n + 1)),
            "tkout_time": pd.date_range("2026-09-01", periods=n, freq="h"),
            "probe_card_id": ["P"] * n, "eqp_id": ["E"] * n, "lot_type": ["P"] * n,
            "rw_cnt": [0] * n}
    return pd.DataFrame({**base, **items})


def test_negative_vr_qa_values_become_positive():
    df = _trend(VR_QA_LEAK=[-1.5, 2.0, -0.0, np.nan], item1=[-1.0, -2.0, 3.0, 4.0])
    got = app.abs_vr_qa(df)
    assert got["VR_QA_LEAK"].tolist()[:3] == [1.5, 2.0, 0.0]
    assert np.isnan(got["VR_QA_LEAK"].tolist()[3]), "빈 값은 빈 값으로"
    assert got["item1"].tolist() == [-1.0, -2.0, 3.0, 4.0], "다른 item 은 그대로"
    assert pd.api.types.is_float_dtype(got["VR_QA_LEAK"])


def test_the_original_frame_is_left_alone():
    df = _trend(VR_QA_1=[-1.0, 2.0])
    app.abs_vr_qa(df)
    assert df["VR_QA_1"].tolist() == [-1.0, 2.0]


def test_text_and_decimal_values_are_read_as_numbers():
    """BigQuery NUMERIC 은 글자나 Decimal 로 올 때가 있다."""
    df = _trend(VR_QA_1=["-3.25", Decimal("-1.5"), "abc", None])
    got = app.abs_vr_qa(df)["VR_QA_1"].tolist()
    assert got[0] == 3.25 and got[1] == 1.5
    assert got[2] == "abc" and got[3] is None, "숫자로 못 읽는 값은 그대로"


def test_spelling_variants_count_as_vr_qa():
    df = _trend(**{"vr-qa 2": [-2.0], "ＶＲ＿ＱＡ3": [-3.0], "xVR_QA": [-4.0]})
    got = app.abs_vr_qa(df)
    assert got["vr-qa 2"].tolist() == [2.0]
    assert got["ＶＲ＿ＱＡ3"].tolist() == [3.0]
    assert got["xVR_QA"].tolist() == [-4.0], "VR_QA 로 '시작' 하는 것만"


def test_limit_columns_next_to_an_item_are_not_touched():
    """예전 구조의 trend 는 item 옆에 관리선 칸을 붙여 온다. 관리선은 값이 아니다."""
    df = _trend(VR_QA_1=[-1.0], VR_QA_1_lsl=[-5.0])
    got = app.abs_vr_qa(df)
    assert got["VR_QA_1"].tolist() == [1.0]
    assert got["VR_QA_1_lsl"].tolist() == [-5.0]


def test_frames_by_product_hands_out_absolute_values():
    frames = list(app.pull_data())
    frames[3] = _trend(VR_QA_1=[-1.0, -2.0])               # ULY trend
    _dc, trend, _spec, _split = app.frames_by_product(frames)
    assert trend["ULY"]["VR_QA_1"].tolist() == [1.0, 2.0]

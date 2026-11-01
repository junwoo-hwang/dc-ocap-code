"""hold 현황 리스트의 lot_type 칸.

hold(dc) 데이터에는 lot_type 이 없고 trend 에는 wafer 마다 있다. 리포트를
만들 때 hold 된 wafer 의 lot_type 을 trend 에서 찾아 붙인다.
"""
import pandas as pd

import app


def _dc(*rows):
    return pd.DataFrame([{"lot_id": f"{root}.1", "root_lot_id": root, "wafer_id": wf,
                          "rw_cnt": rw} for root, wf, rw in rows])


def _trend(*rows):
    return pd.DataFrame([{"root_lot_id": root, "wafer_id": wf, "rw_cnt": rw,
                          "lot_type": lt} for root, wf, rw, lt in rows])


def test_the_held_wafer_gets_its_own_lot_type():
    dc = _dc(("L1", 3, 0), ("L1", 3, 1))
    trend = _trend(("L1", 3, 0, "MP"), ("L1", 3, 1, "ENG"))
    assert app._dc_for_export(dc, trend)["lot_type"].tolist() == ["MP", "ENG"]


def test_lot_and_wafer_are_matched_loosely():
    """dc 는 '03', ' L1 ' 처럼 오고 trend 는 3, 'L1' 로 올 때가 있다."""
    dc = _dc((" L1 ", "03", "0"))
    trend = _trend(("L1", 3, 0, "MP"))
    assert app._dc_for_export(dc, trend)["lot_type"].tolist() == ["MP"]


def test_a_wafer_missing_from_trend_takes_its_lots_usual_lot_type():
    dc = _dc(("L1", 9, 0))
    trend = _trend(("L1", 1, 0, "MP"), ("L1", 2, 0, "MP"), ("L1", 3, 0, "ENG"))
    assert app._dc_for_export(dc, trend)["lot_type"].tolist() == ["MP"]


def test_a_lot_missing_from_trend_is_blank():
    dc = _dc(("L9", 1, 0))
    trend = _trend(("L1", 1, 0, "MP"))
    assert app._dc_for_export(dc, trend)["lot_type"].tolist() == [None]


def test_blank_lot_types_in_trend_are_skipped():
    dc = _dc(("L1", 1, 0))
    trend = _trend(("L1", 1, 0, None), ("L1", 2, 0, " "), ("L1", 3, 0, "PP"))
    assert app._dc_for_export(dc, trend)["lot_type"].tolist() == ["PP"]


def test_a_lot_type_already_in_dc_is_kept():
    dc = _dc(("L1", 1, 0)).assign(lot_type="원본")
    trend = _trend(("L1", 1, 0, "MP"))
    assert app._dc_for_export(dc, trend)["lot_type"].tolist() == ["원본"]


def test_no_lot_type_in_trend_gives_a_blank_column():
    dc = _dc(("L1", 1, 0))
    trend = _trend(("L1", 1, 0, "MP")).drop(columns="lot_type")
    assert app._dc_for_export(dc, trend)["lot_type"].tolist() == [None]


def test_the_original_dc_frame_is_left_alone():
    dc = _dc(("L1", 1, 0))
    app._dc_for_export(dc, _trend(("L1", 1, 0, "MP")))
    assert "lot_type" not in dc.columns


def test_every_mock_hold_finds_a_lot_type():
    dc, trend, _spec, _split = app.frames_by_product(app.pull_data())
    for p in dc:
        got = app._dc_for_export(dc[p], trend[p])
        assert got["lot_type"].notna().all(), p

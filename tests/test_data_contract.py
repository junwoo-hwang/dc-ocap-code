"""pull_data() 가 돌려준 것을 화면이 어떻게 받아들이는가.

여기서 잡으려는 것은 '조용히 틀리는' 부류다 -- 오류 없이 빈 화면이나
빠진 타점으로 나타나서, 데이터를 아는 사람만 알아챌 수 있는 것들.
"""
import numpy as np
import pandas as pd
import pytest

import app


@pytest.fixture(scope="module")
def frames():
    return app.frames_by_product(app.pull_data())


# ------------------------------------------------ pull_data 의 순서 계약

def test_pull_data_returns_twelve_frames_in_the_documented_order():
    frames = app.pull_data()
    assert len(frames) == 12, "순서가 계약이다 -- 개수가 바뀌면 계약도 바뀐 것"
    dc, trend, spec, split = app.frames_by_product(frames)
    for name, d in (("dc", dc), ("trend", trend), ("spec", spec), ("split", split)):
        assert set(d) == {"ULY", "TTS", "SOL"}, name
    # 제품이 서로 뒤바뀌지 않았는가 (뒤바뀌어도 화면엔 오류가 안 뜬다)
    for product, cfg in app.PRODUCT_CONFIG.items():
        assert set(dc[product]["process_id"]) == {cfg["process_id"]}
        assert set(split[product]["process_id"]) == {cfg["process_id"]}


def test_frames_by_product_ignores_the_tail_load_data_appends(frames):
    padded = app.pull_data() + ("2026/01/01 00:00", [], [])
    assert app.frames_by_product(padded)[0].keys() == frames[0].keys()


# ------------------------------------------------------------- check_data

def test_check_data_is_clean_on_the_mock(frames):
    dc, trend, spec, split = frames
    problems, _warnings = app.check_data(dc, trend, spec, split)
    assert problems == []


def test_check_data_reports_a_missing_split_column_as_a_warning(frames):
    """split 이 망가져도 나머지 대시보드는 돈다 -- 막지 말고 알려만 준다."""
    dc, trend, spec, split = frames
    broken = {p: df.drop(columns=["title", "reason"]) for p, df in split.items()}
    problems, warnings = app.check_data(dc, trend, spec, broken)
    assert problems == [], "split 문제로 대시보드 전체를 막으면 안 된다"
    assert any("title" in w and "_split" in w for w in warnings), warnings


def test_check_data_notices_split_that_matches_no_lot(frames):
    """process_id 를 잘못 잘라 다른 제품 것을 넣은 경우.

    이러면 EINECN 팝업이 늘 비어 있는데 화면에는 아무 오류도 안 뜬다.
    """
    dc, trend, spec, split = frames
    swapped = dict(split)
    swapped["ULY"] = split["TTS"]
    _problems, warnings = app.check_data(dc, trend, spec, swapped)
    assert any("겹치지 않습니다" in w for w in warnings), warnings


def test_check_data_without_split_says_nothing_about_it(frames):
    dc, trend, spec, _split = frames
    _problems, warnings = app.check_data(dc, trend, spec)
    assert not any("_split" in w for w in warnings)


def test_check_data_blocks_a_wiring_mistake(frames):
    """dc 와 trend 가 하나도 안 맞으면 그리기 전에 막아야 한다."""
    dc, trend, spec, split = frames
    wrong = {p: df.assign(root_lot_id="NOSUCHLOT") for p, df in dc.items()}
    problems, _warnings = app.check_data(wrong, trend, spec, split)
    assert any("하나도 매칭되지 않습니다" in p for p in problems), problems


# ------------------------------------------------------- split -> 팝업

def test_build_error_is_an_ordinary_exception():
    """SystemExit 였다면 except Exception 으로 못 잡아, 이 함수를 import 해서
    쓰는 쪽(스케줄러 스크립트, 노트북, 테스트)의 프로세스가 통째로 죽는다."""
    assert issubclass(app.BuildError, Exception)
    assert not issubclass(app.BuildError, SystemExit)


def test_split_export_raises_a_catchable_error_not_systemexit(frames):
    """import 해서 쓰는 쪽의 프로세스를 죽이면 안 된다."""
    split = frames[3]["ULY"].drop(columns=["step_desc"])
    try:
        app._split_for_export(split)
    except Exception as err:                 # SystemExit 면 여기 안 걸린다
        assert isinstance(err, app.BuildError)
        assert "step_desc" in str(err)
    else:
        pytest.fail("칸이 빠졌는데 그냥 통과했습니다")


def test_split_export_keeps_rows_that_differ_only_in_wafers(frames):
    """설비가 달라 나뉜 줄은 합치면 안 된다 -- 합치면 split 이 사라진다."""
    base = frames[3]["ULY"].iloc[0].copy()
    a, b = base.copy(), base.copy()
    for col in app.SPLIT_WAFER_COLUMNS:
        a[col] = "V" if col in ("2", "3") else ""
        b[col] = "V" if col in ("10", "11") else ""
    out = app._split_for_export(pd.DataFrame([a, b]))
    assert len(out) == 2, "comp_id_list 만 다른 줄은 각각 남아야 한다"


def test_split_export_orders_tests_by_their_first_step(frames):
    """팝업이 test 단위로 칸을 합치므로 같은 test 의 줄이 흩어지면 안 된다."""
    out = app._split_for_export(frames[3]["ULY"])
    for lot, chunk in out.groupby("root_lot_id", sort=False):
        seen, first_steps = [], []
        for einecn, part in chunk.groupby("einecn_no", sort=False):
            assert einecn not in seen, f"{lot}/{einecn} 의 줄이 흩어져 있다"
            seen.append(einecn)
            first_steps.append(part["step_seq"].min())
            assert list(part["step_seq"]) == sorted(part["step_seq"])
        assert first_steps == sorted(first_steps), f"{lot}: test 순서가 step 순이 아니다"


def test_split_export_drops_only_unreachable_lots(frames):
    dc, trend, _spec, split = frames
    keep = app.reachable_lots(dc["ULY"], trend["ULY"])
    out = app._split_for_export(split["ULY"], keep)
    assert len(out) <= len(split["ULY"])
    assert set(out["root_lot_id"].map(app.norm_lot)) <= keep
    # 화면에서 고를 수 있는 lot 은 하나도 빠지면 안 된다
    reachable_rows = split["ULY"][split["ULY"]["root_lot_id"].map(app.norm_lot).isin(keep)]
    assert len(out) == len(reachable_rows)


def test_reachable_lots_covers_chart_clicks_not_just_holds(frames):
    """차트의 회색 타점도 누를 수 있으므로 trend 의 lot 까지 열릴 수 있다."""
    dc, trend, _spec, _split = frames
    keep = app.reachable_lots(dc["ULY"], trend["ULY"])
    assert set(trend["ULY"]["root_lot_id"].map(app.norm_lot)) <= keep
    assert set(dc["ULY"]["root_lot_id"].map(app.norm_lot)) <= keep


# ------------------------------------------------------------- 목록 로직

def test_hold_view_excludes_lots_that_already_flowed(frames):
    """조치 기록은 다음날 아침에 오지만 상태는 자주 갱신된다.

    그래서 '조치 기록이 비었다' 만으로 hold 를 판정하면, 이미 흘려보낸
    lot 이 hold 목록에 하루 더 남는다.
    """
    dc = frames[0]["ULY"].copy()
    dc.loc[:, ["owner", "code", "comment"]] = None
    dc.loc[:, "status"] = "Run"
    assert app.filter_by_status(dc, "hold").empty


def test_group_holds_keeps_a_rework_separate(frames):
    """같은 lot_id 라도 rw_cnt 가 다르면 다른 건이다."""
    dc = frames[0]["ULY"]
    grouped = app.group_holds(dc)
    assert not grouped.duplicated(["lot_id", "rw_cnt"]).any()
    reworked = dc[dc["rw_cnt"] > 0]
    if not reworked.empty:
        lot = reworked.iloc[0]["lot_id"]
        assert (grouped["lot_id"] == lot).sum() >= 2


def test_limits_asof_uses_the_revision_live_at_that_moment(frames):
    """규격이 바뀐 날 앞뒤로 다른 값이 나와야 한다 (차트의 계단)."""
    spec_rows = app.item_spec_rows(frames[2]["ULY"], "item1")
    if len(spec_rows) < 2:
        pytest.skip("이 item 은 개정 이력이 하나뿐")
    change = spec_rows["from_time"].iloc[1]
    times = pd.Series([change - pd.Timedelta(days=1), change + pd.Timedelta(days=1)])
    lim = app.limits_asof(spec_rows, times)
    assert lim["ucl"].iloc[0] != lim["ucl"].iloc[1]


# --------------------------------------------------------------- 빌드

def test_build_writes_a_page_carrying_the_shared_constants(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_PATH", tmp_path / "out.html")
    out = app.build_dc_ocap_html()
    html = out.read_text(encoding="utf-8")
    assert "const SHARED = " in html
    assert f'"wacMaxGray":{app.WAC_MAX_GRAY}' in html
    assert "__DATA_B64__" not in html and "/*__SHARED_CONSTANTS__*/" not in html
    assert out.stat().st_size > 500_000       # plotly 가 통째로 들어 있다


def test_build_refuses_a_template_missing_a_placeholder(tmp_path, monkeypatch):
    """app.py 와 템플릿의 버전이 어긋나면 빈 리포트를 내보내지 말고 멈춘다."""
    stale = tmp_path / "stale.html"
    stale.write_text(
        app.TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("/*__SHARED_CONSTANTS__*/", ""), encoding="utf-8")
    monkeypatch.setattr(app, "TEMPLATE_PATH", stale)
    monkeypatch.setattr(app, "OUTPUT_PATH", tmp_path / "out.html")
    with pytest.raises(app.BuildError) as err:
        app.build_dc_ocap_html()
    assert "__SHARED_CONSTANTS__" in str(err.value)

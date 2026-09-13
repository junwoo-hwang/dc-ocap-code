"""pull_data() 가 돌려준 것을 화면이 어떻게 받아들이는가.

여기서 잡으려는 것은 '조용히 틀리는' 부류다 -- 오류 없이 빈 화면이나
빠진 타점으로 나타나서, 데이터를 아는 사람만 알아챌 수 있는 것들.
"""
import base64
import gzip
import json
import re

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


# --------------------------------------- JSON 으로 내보낼 수 있는 값인가

def test_infinity_is_sent_as_missing_not_as_infinity(frames):
    """inf 셀 하나가 리포트 전체를 못 열게 만들던 것.

    json.dumps 는 inf 를 `Infinity` 라고 적는데 그건 JSON 이 아니라서,
    브라우저의 JSON.parse 가 거기서 멈춘다 -- 차트 하나가 아니라 페이지가
    통째로 "데이터를 읽는 중 오류" 한 줄이 된다.
    """
    assert app._clean(float("inf")) is None
    assert app._clean(float("-inf")) is None
    assert app._clean(np.float64("inf")) is None
    assert app._clean(1.5) == 1.5 and app._clean(0.0) == 0.0

    trend = frames[1]["ULY"].copy()
    item = app.item_columns(trend)[0]
    trend.loc[trend.index[0], item] = float("inf")
    dumped = json.dumps(app._columns(trend))
    assert "Infinity" not in dumped
    json.loads(dumped)                     # 브라우저가 하는 일과 같다


def test_duplicate_columns_are_refused_instead_of_exported_as_their_own_names(frames):
    """df["a"] 가 칸 두 개를 가리키면 Series 가 아니라 DataFrame 이라,
    값 대신 칸 이름이 실린 리포트가 오류 없이 만들어진다."""
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])
    with pytest.raises(app.BuildError) as err:
        app._columns(df)
    assert "a" in str(err.value)

    dc, trend, spec, split = frames
    doubled = dict(trend)
    item = app.item_columns(trend["ULY"])[0]
    doubled["ULY"] = pd.concat([trend["ULY"], trend["ULY"][[item]]], axis=1)
    problems, _warnings = app.check_data(dc, doubled, spec, split)
    assert any("겹치는 칸" in p for p in problems), problems


# ------------------------------------------------------- 정규화 규칙

@pytest.mark.parametrize("value", [3, 3.0, "3", "03", " 3 ", "3.0"])
def test_wafer_ids_normalize_to_the_same_number_whatever_the_dtype(value):
    """trend 는 wafer_id 를 float 로, dc 는 int 로 주는 일이 흔하다.

    예전에는 3.0 만 문자열 "3.0" 으로 남아서, 그 둘이 한 쌍도 안 맞고
    check_data 가 "하나도 매칭되지 않습니다" 로 빌드를 멈춰 세웠다.
    """
    assert app.norm_wafer(value) == 3


def test_a_float_wafer_column_does_not_block_the_build(frames):
    dc, trend, spec, split = frames
    floated = dict(trend)
    floated["ULY"] = trend["ULY"].assign(
        wafer_id=pd.to_numeric(trend["ULY"]["wafer_id"], errors="coerce").astype(float))
    problems, _warnings = app.check_data(dc, floated, spec, split)
    assert not any("매칭되지 않습니다" in p for p in problems), problems


def test_limit_columns_left_in_trend_are_not_counted_as_items():
    """예전 구조(trend 안에 item1_ucl) 로 뽑힌 trend 가 섞여 들어와도
    차트가 다섯 배로 늘지 않아야 한다."""
    cols = app.META_TREND_COLS + ["item1", "item1_ucl", "item1_lcl",
                                  "item1_usl", "item1_lsl", "leak_usl"]
    got = app.item_columns(pd.DataFrame(columns=cols))
    assert got == ["item1", "leak_usl"], (
        "item1_* 는 빼되, 짝이 되는 item 이 없는 leak_usl 은 진짜 item 이다"
    )


# ------------------------------------------------- 자리표시자 관리선
# 규격이 아직 없는 item 에 usl=99999 가 들어오면 y축이 100k 까지 늘어나
# 나머지 타점 전부가 바닥에 한 줄로 눌린다. 오류는 하나도 안 뜬다.

@pytest.mark.parametrize("sentinel", [99999, 999999.0, 9999, -99999, float("inf")])
def test_nine_placeholders_are_dropped_whatever_the_item_size(frames, sentinel):
    """9 로만 된 값은 item 값의 크기와 상관없이 관리선이 아니다."""
    trend = pd.DataFrame({"item1": [1e3, 1.2e3, 1.1e3]})     # 값 자체가 큰 item
    spec = pd.DataFrame({"item_id": ["item1"], "from_time": ["2026-01-01"],
                         "ucl": [1300.0], "lcl": [900.0],
                         "usl": [float(sentinel)], "lsl": [800.0]})
    out, dropped = app.drop_placeholder_limits(spec, trend)
    assert pd.isna(out["usl"].iloc[0])
    assert out["ucl"].iloc[0] == 1300.0, "멀쩡한 관리선까지 지우면 안 된다"
    assert [d[1] for d in dropped] == ["usl"]


def test_an_unusual_placeholder_is_caught_by_the_data_it_dwarfs():
    """88888 처럼 9 가 아닌 자리표시자는 측정값을 잣대로 잡는다."""
    trend = pd.DataFrame({"item1": [0.30, 0.55, 0.70]})
    spec = pd.DataFrame({"item_id": ["item1"], "from_time": ["2026-01-01"],
                         "ucl": [0.8], "lcl": [0.2], "usl": [88888.0], "lsl": [0.1]})
    out, _ = app.drop_placeholder_limits(spec, trend)
    assert pd.isna(out["usl"].iloc[0])
    assert out["ucl"].iloc[0] == 0.8


def test_a_generous_but_real_limit_survives():
    """규격이 넉넉한 item 을 자리표시자로 오해하면 안 된다.

    측정값의 1000배까지는 남긴다 -- 잘못 지우는 쪽이 더 위험하다.
    """
    trend = pd.DataFrame({"item1": [100.0, 150.0, 200.0]})
    spec = pd.DataFrame({"item_id": ["item1"], "from_time": ["2026-01-01"],
                         "ucl": [250.0], "lcl": [50.0], "usl": [20000.0], "lsl": [0.0]})
    out, dropped = app.drop_placeholder_limits(spec, trend)
    assert out["usl"].iloc[0] == 20000.0, dropped
    assert dropped == []


def test_without_trend_only_the_nine_rule_applies():
    """잣대가 없으면 큰 수를 함부로 지우지 않는다."""
    spec = pd.DataFrame({"item_id": ["item1", "item1"],
                         "from_time": ["2026-01-01", "2026-02-01"],
                         "ucl": [1.0, 1.0], "lcl": [0.0, 0.0],
                         "usl": [99999.0, 88888.0], "lsl": [0.0, 0.0]})
    out, _ = app.drop_placeholder_limits(spec, None)
    assert pd.isna(out["usl"].iloc[0])
    assert out["usl"].iloc[1] == 88888.0


def test_placeholder_limits_never_reach_the_browser(frames):
    """_spec_for_export 가 실제로 빼는가 -- 여기가 빠지면 화면이 그대로 눌린다."""
    dc, trend, spec, _split = frames
    poisoned = spec["ULY"].copy()
    item = str(poisoned["item_id"].iloc[0])
    poisoned.loc[poisoned.index[0], "usl"] = 99999.0
    out = app._spec_for_export(poisoned, trend["ULY"])
    assert out["usl"].max() < 99999.0
    # 잣대를 안 넘겨주면 못 뺀다는 것도 같이 못박아 둔다
    assert app._spec_for_export(poisoned)["usl"].max() < 99999.0   # 9-규칙으로 잡힘
    assert item


def test_the_built_page_carries_no_placeholder_limit(tmp_path, monkeypatch, frames):
    """빌드가 잣대(trend)를 안 넘겨주면 9 가 아닌 자리표시자는 그대로 실린다.

    이 검사가 없으면 drop_placeholder_limits 가 아무리 맞아도 화면은
    그대로 눌린 채였다 -- 부르는 쪽에서 인자 하나만 빠지면 되니까.
    """
    dc, trend, spec, split = frames
    item = str(spec["ULY"]["item_id"].iloc[0])
    ceiling = pd.to_numeric(trend["ULY"][item], errors="coerce").abs().max()
    sentinel = float(round(ceiling * 5000))       # 9 로만 된 수가 아니다
    bad = spec["ULY"].copy()
    bad.loc[bad.index[0], "usl"] = sentinel

    real = app.pull_data

    def poisoned():
        out = list(real())
        out[6] = bad                              # uly_spec 자리
        return tuple(out)

    monkeypatch.setattr(app, "pull_data", poisoned)
    monkeypatch.setattr(app, "OUTPUT_PATH", tmp_path / "out.html")
    html = app.build_dc_ocap_html().read_text(encoding="utf-8")

    b64 = re.search(r'const DATA_B64 = "([^"]*)"', html).group(1)
    payload = json.loads(gzip.decompress(base64.b64decode(b64)))
    assert sentinel not in payload["spec"]["ULY"]["usl"], (
        f"{sentinel} 가 그대로 실렸습니다. y축이 거기까지 늘어납니다."
    )


def test_check_data_names_the_item_whose_limit_was_dropped(frames):
    dc, trend, spec, split = frames
    poisoned = dict(spec)
    bad = spec["ULY"].copy()
    bad.loc[bad.index[0], "usl"] = 99999.0
    poisoned["ULY"] = bad
    problems, warnings = app.check_data(dc, trend, poisoned, split)
    assert problems == [], "관리선 하나 때문에 리포트를 막으면 안 된다"
    hit = [w for w in warnings if "자리표시자" in w or "동떨어진" in w]
    assert hit, warnings
    assert str(bad["item_id"].iloc[0]) in hit[0] and "usl" in hit[0]


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


def test_a_lower_rw_cnt_moves_to_the_history_list(frames):
    """rework 로 다시 걸리면 앞 건은 끝난 것이다.

    이 규칙이 브라우저에만 있어서, diagnose.py 가 알려주는 hold 건수가
    화면에 보이는 것보다 많았다.
    """
    dc = frames[0]["ULY"].iloc[:1].copy()
    base = dc.iloc[0].to_dict()
    rows = []
    for rw in (0, 1, 2):
        r = dict(base, rw_cnt=rw, code=None, owner=None, comment=None, status="Hold")
        rows.append(r)
    dc = pd.DataFrame(rows)
    held = app.filter_by_status(dc, "hold")
    assert list(held["rw_cnt"]) == [2], "가장 높은 rw_cnt 만 열려 있어야 한다"
    assert sorted(app.filter_by_status(dc, "이력")["rw_cnt"]) == [0, 1]


def test_an_unreadable_rw_cnt_is_neither_pushed_nor_pushes(frames):
    """rw_cnt 를 못 읽는 줄은 순서를 매길 수 없으니 건드리지 않는다."""
    base = frames[0]["ULY"].iloc[0].to_dict()
    dc = pd.DataFrame([
        dict(base, rw_cnt=None, code=None, owner=None, status="Hold"),
        dict(base, rw_cnt=5, code=None, owner=None, status="Hold"),
    ])
    assert len(app.filter_by_status(dc, "hold")) == 2


def test_one_dispositioned_item_does_not_split_an_event_across_both_lists(frames):
    """한 건 안에서 item 별로 조치가 갈려도 hold 와 이력 양쪽에 뜨면 안 된다."""
    base = frames[0]["ULY"].iloc[0].to_dict()
    dc = pd.DataFrame([
        dict(base, rw_cnt=0, item_id="item1", code=None, owner=None, status="Hold"),
        dict(base, rw_cnt=0, item_id="item2", code="C1", owner="kim", status="Hold"),
    ])
    hold, hist = app.filter_by_status(dc, "hold"), app.filter_by_status(dc, "이력")
    assert len(hold) == 2 and len(hist) == 0


def test_group_holds_keeps_a_rework_separate(frames):
    """같은 lot_id 라도 rw_cnt 가 다르면 다른 건이다."""
    dc = frames[0]["ULY"]
    grouped = app.group_holds(dc)
    assert not grouped.duplicated(["lot_id", "rw_cnt"]).any()
    reworked = dc[dc["rw_cnt"] > 0]
    if not reworked.empty:
        lot = reworked.iloc[0]["lot_id"]
        assert (grouped["lot_id"] == lot).sum() >= 2


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

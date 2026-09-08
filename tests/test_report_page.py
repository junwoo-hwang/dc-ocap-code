"""리포트를 실제 브라우저에서 열어, 자바스크립트에만 있는 규칙을 확인한다.

차트 로직은 전부 dc_ocap_template.html 안에 있어서 파이썬 테스트로는 닿지
않는다. 그래서 여기만 브라우저를 띄운다 -- playwright 가 없으면 통째로
건너뛰므로, 나머지 테스트는 브라우저 없이도 돈다.

    pip install playwright && playwright install chromium
"""
import os

import numpy as np
import pytest

import app

sync_playwright = pytest.importorskip(
    "playwright.sync_api", reason="playwright 가 없으면 브라우저 검사는 건너뛴다"
).sync_playwright

SENTINEL = 9.9e9
N_ZEROS = 12
N_ROWS = 4000          # WAC_MAX_GRAY 보다 많아야 솎는 게 관찰된다


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    """일부러 망가뜨린 값을 심은 리포트를 한 번만 만든다.

    - item_A: hold 안 걸린 극단값 하나 + hold 걸린 극단값 하나
    - item_B: 0 근처에 몰린 값과 정확히 0 인 값 여러 개
    """
    out = tmp_path_factory.mktemp("report") / "dc_ocap.html"
    orig_probe, orig_pull = app.generate_probe_df, app.pull_data
    planted = {}

    def big_probe(product, n_rows=300):
        return orig_probe(product, n_rows=N_ROWS if product == "ULY" else n_rows)

    def doctored_pull():
        app.generate_probe_df = big_probe
        try:
            frames = list(orig_pull())
        finally:
            app.generate_probe_df = orig_probe
        dc, trend = frames[0], frames[3].reset_index(drop=True)

        # hold 가 걸린 (lot, wafer, item) 하나를 고른다
        held = dc.iloc[0]
        item_a = app.resolve_item_col(trend, held["item_id"])
        same = trend[(trend["root_lot_id"].map(app.norm_lot) == app.norm_lot(held["root_lot_id"]))
                     & (trend["wafer_id"].map(app.norm_wafer) == app.norm_wafer(held["wafer_id"]))]
        assert len(same), "hold 건이 trend 에 없다"
        trend.loc[same.index[0], item_a] = SENTINEL          # hold 됨 -> 남아야 한다

        held_pairs = set(zip(dc["root_lot_id"].map(app.norm_lot),
                             dc["wafer_id"].map(app.norm_wafer)))
        free = trend[[(l, w) not in held_pairs for l, w in
                      zip(trend["root_lot_id"].map(app.norm_lot),
                          trend["wafer_id"].map(app.norm_wafer))]]
        trend.loc[free.index[0], item_a] = SENTINEL          # hold 아님 -> 빠져야 한다

        # 0 근처에 몰린 item (누설 같은 것). 정확히 0 이 흔하다.
        item_b = next(c for c in app.item_columns(trend) if c != item_a)
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 0.2, len(trend)).round(4)
        # 반올림으로 우연히 0.0 이 나오면 아래에서 심은 0 과 구분이 안 된다
        trend[item_b] = np.where(noise == 0.0, 0.01, noise)
        trend.loc[trend.index[:N_ZEROS], item_b] = 0.0

        item_c = next(c for c in app.item_columns(trend) if c not in (item_a, item_b))
        planted.update(item_a=item_a, item_b=item_b, item_c=item_c)
        frames[3] = trend
        return tuple(frames)

    app.pull_data = doctored_pull
    old_out = app.OUTPUT_PATH
    app.OUTPUT_PATH = out
    try:
        app.build_dc_ocap_html()
    finally:
        app.pull_data, app.OUTPUT_PATH = orig_pull, old_out
    return out, planted


@pytest.fixture(scope="module")
def wac_page(report):
    """WAC 페이지를 띄우고, item 이름으로 차트의 y 값을 읽을 수 있게 해준다."""
    path, planted = report
    exe = os.environ.get("PLAYWRIGHT_CHROMIUM", "/opt/pw-browsers/chromium")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=exe if os.path.exists(exe) else None)
        page = browser.new_page(viewport={"width": 1500, "height": 950})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(path.as_uri())
        page.wait_for_selector("#listBody tr", timeout=60_000)
        page.click('#pageSwitch button[data-val="wac"]')
        page.wait_for_timeout(3000)

        def traces_for(item):
            """그 item 차트의 트레이스별 y 값 (이름 -> 리스트)."""
            page.fill("#wacSearch", item)
            page.wait_for_timeout(2500)
            return page.evaluate(
                """(want) => {
                  for (const box of document.querySelectorAll('.wac-chart')) {
                    const title = box.querySelector('.wac-chart-title');
                    if (!title || title.textContent.trim() !== want) continue;
                    const g = box.querySelector('div.js-plotly-plot');
                    if (!g || !g.data) return null;
                    const out = {};
                    for (const t of g.data) {
                      if (!t.name || !t.y) continue;
                      out[t.name] = (out[t.name] || []).concat(Array.from(t.y));
                    }
                    return out;
                  }
                  return null;
                }""", item)

        yield page, planted, traces_for, errors
        browser.close()


def test_page_loads_without_console_errors(wac_page):
    _page, _planted, _traces, errors = wac_page
    assert errors == []


def test_shared_constants_reach_the_browser(wac_page):
    page, _planted, _traces, _errors = wac_page
    got = page.evaluate("() => ({ maxGray: WAC_MAX_GRAY, k: OUTLIER_K,"
                        " minN: OUTLIER_MIN_N, wafers: EIN_WAFERS.length })")
    assert got == {"maxGray": app.WAC_MAX_GRAY, "k": app.OUTLIER_K,
                   "minN": app.OUTLIER_MIN_N, "wafers": len(app.SPLIT_WAFER_COLUMNS)}


def test_an_absurd_reading_is_dropped_unless_the_wafer_was_held(wac_page):
    """5시그마 밖 값은 빼되, hold 걸린 wafer 는 남긴다.

    hold 사유가 바로 그 값인 경우가 많아서, 숨기면 왜 걸렸는지가 사라진다.
    """
    _page, planted, traces_for, _errors = wac_page
    data = traces_for(planted["item_a"])
    assert data, f"{planted['item_a']} 차트를 못 찾았습니다"
    drawn = [y for ys in data.values() for y in ys if y is not None]
    assert sum(1 for y in drawn if y > 1e8) == 1, "hold 건 하나만 남아야 합니다"


def test_exact_zeros_survive_on_an_item_centred_on_zero(wac_page):
    """예전에 '정확히 0 = 계측 실패' 로 지우던 규칙의 회귀 검사.

    누설처럼 값이 0 근처인 item 에서는 진짜 0 이 흔하다. 지우면 멀쩡한
    타점이 소리 없이 사라진다.
    """
    _page, planted, traces_for, _errors = wac_page
    data = traces_for(planted["item_b"])
    assert data, f"{planted['item_b']} 차트를 못 찾았습니다"
    zeros = sum(1 for ys in data.values() for y in ys if y == 0.0)
    assert zeros == N_ZEROS, f"0 이 {N_ZEROS}개여야 하는데 {zeros}개"


def test_only_the_background_is_thinned(wac_page):
    """회색은 솎되 CL OUT / SL OUT 은 하나도 솎지 않는다 -- 그게 신호다."""
    _page, planted, traces_for, _errors = wac_page
    data = traces_for(planted["item_c"])
    assert data, f"{planted['item_c']} 차트를 못 찾았습니다"
    # 상한 + 1 인 이유: 고르게 솎으면 마지막 점에 못 닿아서 차트가 실제보다
    # 일찍 끝난 것처럼 보이므로, 끝 점은 따로 넣는다 (양쪽 구현 다 그렇다)
    assert len(data.get("trend", [])) <= app.WAC_MAX_GRAY + 1
    assert sum(len(v) for v in data.values()) > app.WAC_MAX_GRAY, (
        "솎는 게 관찰되려면 원본이 상한보다 많아야 한다"
    )

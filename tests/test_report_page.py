"""리포트를 실제 브라우저에서 열어, 자바스크립트에만 있는 규칙을 확인한다.

차트 로직은 전부 dc_ocap_template.html 안에 있어서 파이썬 테스트로는 닿지
않는다. 그래서 여기만 브라우저를 띄운다 -- playwright 가 없으면 통째로
건너뛰므로, 나머지 테스트는 브라우저 없이도 돈다.

    pip install playwright && playwright install chromium
"""
import json
import os

import numpy as np
import pytest

import app

sync_playwright = pytest.importorskip(
    "playwright.sync_api", reason="playwright 가 없으면 브라우저 검사는 건너뛴다"
).sync_playwright

SENTINEL = 9.9e9
PLACEHOLDER_LIMIT = 99999.0    # 규격 미정 자리표시자로 들어오는 관리선
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

        # 규격이 아직 없는 item 에 usl=99999 가 들어온 경우. 그대로 그리면
        # y축이 100k 까지 늘어나 나머지 타점이 바닥에 한 줄로 눌린다.
        item_d = next(c for c in app.item_columns(trend)
                      if c not in (item_a, item_b, item_c))
        spec = frames[6].copy()
        hit = (spec["item_id"].astype(str).str.strip().str.lower()
               == str(item_d).strip().lower())
        assert hit.any(), f"spec 에 {item_d} 가 없다"
        spec.loc[hit, "usl"] = PLACEHOLDER_LIMIT
        frames[6] = spec

        planted.update(item_a=item_a, item_b=item_b, item_c=item_c, item_d=item_d)
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


@pytest.fixture(autouse=True)
def _reset(wac_page):
    """페이지 하나를 여러 테스트가 나눠 쓰므로, 매번 같은 자리에서 시작한다.

    (제품을 바꾸거나 검색을 걸어둔 채로 끝나면 다음 테스트가 다른 화면을
    본다 -- 실제로 그래서 엉뚱한 테스트가 깨졌다.)
    """
    page, _planted, _traces, _errors = wac_page
    page.evaluate("""() => {
      closeWacModal();
      state.product = 'ULY';
      state.wacSelected = null;
      state.wacPerRow = WAC_PER_ROW_DEFAULT;
      document.getElementById('wacSearch').value = '';
      state.wacSearch = '';
      renderAll();
    }""")
    page.wait_for_timeout(1500)
    yield


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


def _hover(page, kind):
    """조치 내역이 있는(kind="dispo") / 없는(kind="none") CL·SL 타점에 hover.

    타점의 화면 좌표와 hover 상자의 사각형을 같이 돌려준다.
    """
    return page.evaluate(
        """(kind) => {
          for (const g of document.querySelectorAll('.wac-chart div.js-plotly-plot')) {
            for (let ti = 0; ti < (g.data||[]).length; ti++) {
              const t = g.data[ti];
              if (t.name !== 'CL OUT' && t.name !== 'SL OUT') continue;
              const want = (cd) => {
                const s = cd[7] || '';
                if (kind === 'dispo') return s.includes('클릭하면');
                if (kind === 'none') return s.includes('hold 이력 없음');
                // 'dash': hold 기록은 있는데 comment 도 owner 도 비어 있는 줄
                return !s.includes('hold 이력 없음')
                       && /comment : -(<br>|$)/.test(s) && s.includes('owner : -');
              };
              const i = (t.customdata||[]).findIndex(want);
              if (i < 0) continue;
              Plotly.Fx.hover(g, [{curveNumber: ti, pointNumber: i}]);
              const lab = g.querySelector('.hoverlayer .hovertext');
              const lb = lab ? lab.getBoundingClientRect() : null;
              const fl = g._fullLayout;
              const bb = g.querySelector('.main-svg').getBoundingClientRect();
              const px = bb.left + fl.xaxis._offset + fl.xaxis.l2p(fl.xaxis.d2c(t.x[i]));
              const py = bb.top + fl.yaxis._offset + fl.yaxis.l2p(fl.yaxis.d2c(t.y[i]));
              g.dataset.hoverTarget = '1';
              return { text: t.customdata[i][7], cd: t.customdata[i].slice(0, 2),
                       item: g.parentElement.querySelector('.wac-chart-title').textContent.trim(),
                       ti, pi: i, point: [px, py],
                       box: lb ? [lb.left, lb.top, lb.right, lb.bottom] : null };
            }
          }
          return null;
        }""", kind)


def _click_hovered(page, hit):
    """_hover 가 짚은 바로 그 타점을 클릭한다.

    (lot, wafer) 로만 찾으면 같은 wafer 가 모든 item 차트에 있어서 엉뚱한
    차트의 타점을 누르게 된다 -- 실제로 그래서 이 테스트가 결함을 놓쳤다.
    """
    ok = page.evaluate(
        """([ti, pi]) => {
          const g = document.querySelector('.wac-chart div.js-plotly-plot[data-hover-target]');
          if (!g) return false;
          g.emit('plotly_click', {points: [{customdata: g.data[ti].customdata[pi]}]});
          return true;
        }""", [hit["ti"], hit["pi"]])
    assert ok, "hover 했던 차트를 다시 못 찾았습니다"
    page.wait_for_timeout(400)


def test_the_hover_box_never_covers_the_point_it_describes(wac_page):
    """가운데 타점은 상자가 놓일 자리가 좁아, 예전에는 상자가 타점을 덮었다."""
    page, _planted, _traces, _errors = wac_page
    page.fill("#wacSearch", "")
    page.wait_for_timeout(2000)
    hit = _hover(page, "dispo") or _hover(page, "none")
    assert hit and hit["box"], "CL/SL 타점을 못 찾았습니다"
    (px, py), (l, t, r, b) = hit["point"], hit["box"]
    covered = l - 2 <= px <= r + 2 and t - 2 <= py <= b + 2
    assert not covered, f"hover 상자 {hit['box']} 가 타점 {hit['point']} 을 덮습니다"


def test_the_hover_comment_is_clipped_to_the_owner_line(wac_page):
    """comment 는 길다. 상자가 커지면 놓을 자리가 없어 타점을 덮으므로,
    owner / code 줄의 폭에서 자르고 전문은 팝업으로 보낸다."""
    page, _planted, _traces, _errors = wac_page
    hit = _hover(page, "dispo")
    if not hit:
        pytest.skip("조치 내역이 있는 CL/SL 타점이 mock 에 없다")
    lines = [l for l in hit["text"].split("<br>") if not l.startswith("<i>")]
    width = page.evaluate("(s) => displayWidth(s)", lines[0])
    cap = max(page.evaluate("(s) => displayWidth(s)", l) for l in lines[1:])
    assert width <= max(cap, 24), f"comment 줄이 {width} 칸, 기준 {cap} 칸"
    assert "클릭하면" in hit["text"], "전문을 어디서 보는지 안내가 없다"


def test_clicking_a_point_with_a_disposition_opens_the_full_text(wac_page):
    page, _planted, _traces, _errors = wac_page
    page.evaluate("() => closeWacModal()")
    hit = _hover(page, "dispo")
    if not hit:
        pytest.skip("조치 내역이 있는 CL/SL 타점이 mock 에 없다")
    _click_hovered(page, hit)
    assert not page.evaluate("() => document.getElementById('wacModal').hidden")
    body = page.inner_text("#wacModalBody")
    assert "comment" in body and "owner" in body and "code" in body
    assert "…" not in body, "팝업에서는 자르지 않는다"
    # 팝업을 닫아도 그 wafer 는 선택된 채로 남는다 (원래 클릭 동작)
    page.evaluate("() => closeWacModal()")
    assert page.inner_text("#wacSelected").strip() != ""


@pytest.mark.parametrize("kind,why", [
    ("none", "hold 기록 자체가 없는 타점"),
    ("dash", "hold 기록은 있지만 comment 도 owner 도 비어 있는 타점"),
])
def test_clicking_a_point_with_no_disposition_just_selects_it(wac_page, kind, why):
    """comment 도 owner 도 없으면 보여줄 게 없으므로 팝업을 띄우지 않는다.

    두 경우를 다 본다. 'hold 기록이 없다' 와 '기록은 있는데 비어 있다' 는
    다른 상태이고, 뒤쪽만 빠뜨리면 '기록이 있으면 무조건 띄운다' 로 바꿔도
    아무 테스트도 안 깨진다.
    """
    page, _planted, _traces, _errors = wac_page
    page.evaluate("() => { closeWacModal(); state.wacSelected = null; wacApplySelection(); }")
    hit = _hover(page, kind)
    if not hit:
        pytest.skip(f"{why} 이 mock 에 없다")
    _click_hovered(page, hit)
    assert page.evaluate("() => document.getElementById('wacModal').hidden"), "팝업이 뜨면 안 된다"
    assert page.inner_text("#wacSelected").strip() != "", "선택은 되어야 한다"


def test_a_retested_wafer_shows_every_hold_event(wac_page):
    """같은 wafer 가 재측정으로 rw_cnt 0, 1 로 두 번 걸리면 둘 다 보여준다.

    건마다 조치가 따로 붙으므로 합치면 한쪽 기록이 사라진다.
    """
    page, _planted, _traces, _errors = wac_page
    found = page.evaluate(
        """() => {
          for (const product of ['ULY', 'TTS', 'SOL']) {
            const seen = {};
            for (const r of (DATA.dc[product] || [])) {
              const k = [normLot(r.root_lot_id), normWafer(r.wafer_id),
                         String(r.item_id).trim().toLowerCase()].join('|');
              (seen[k] = seen[k] || new Set()).add(normRwCnt(r.rw_cnt));
            }
            const k = Object.keys(seen).find(k => seen[k].size > 1);
            if (!k) continue;
            const [lot, wafer, item] = k.split('|');
            state.product = product;
            const records = wacHoldRecords(product, item, lot, wafer);
            openWacDispoModal(item, lot, wafer, records);
            return { n: records.length, rw: records.map(r => normRwCnt(r.rw_cnt)) };
          }
          return null;
        }""")
    if not found:
        pytest.skip("재측정으로 두 번 걸린 wafer 가 mock 에 없다")
    assert found["n"] >= 2 and len(set(found["rw"])) >= 2, found
    assert page.eval_on_selector_all(".dispo-event", "d => d.length") == found["n"]
    page.evaluate("() => closeWacModal()")


def test_a_placeholder_limit_never_stretches_the_y_axis(wac_page):
    """usl=99999 가 선으로 그려지면 축이 100k 로 고정되고 타점이 눌린다.

    파이썬이 그 값을 빼서 보내므로 브라우저에는 애초에 도착하지 않아야
    한다. 여기서 보는 것은 '실제로 축이 멀쩡한가' 하나다.
    """
    page, planted, _traces, _errors = wac_page
    item = planted["item_d"]
    page.fill("#wacSearch", item)
    page.wait_for_timeout(2500)
    info = page.evaluate(
        """(want) => {
          for (const box of document.querySelectorAll('.wac-chart')) {
            const title = box.querySelector('.wac-chart-title');
            if (!title || title.textContent.trim() !== want) continue;
            const g = box.querySelector('div.js-plotly-plot');
            if (!g || !g.data) return null;
            const ys = [];
            for (const t of g.data) if (t.mode === 'markers' && t.y) ys.push(...t.y);
            return { range: g._fullLayout.yaxis.range,
                     lines: g.data.filter(t => t.mode === 'lines').map(t => t.name),
                     top: Math.max(...ys) };
          }
          return null;
        }""", item)
    assert info, f"{item} 차트를 못 찾았습니다"
    assert "USL" not in info["lines"], "자리표시자가 관리선으로 그려졌습니다"
    assert info["range"][1] < PLACEHOLDER_LIMIT / 100, (
        f"y축이 {info['range']} 까지 늘어났습니다 (타점 최대 {info['top']})"
    )


def test_the_trend_chart_click_survives_a_purge(wac_page):
    """trend 컬럼이 없는 item 을 한 번 보면 Plotly.purge 가 돈다.

    purge 는 차트뿐 아니라 그 element 의 plotly 이벤트 발신기까지 지운다.
    "한 번만 건다" 를 플래그로 기억하고 있으면 다시 걸지 않아서, 그 뒤로는
    차트 타점을 눌러도 아무 일도 안 일어난다 (오류조차 안 난다).
    """
    page, _planted, _traces, _errors = wac_page
    page.evaluate("""() => {
      state.page = 'dcocap';
      document.getElementById('pageDcOcap').hidden = false;
      document.getElementById('pageWac').hidden = true;
      state.selectedLotId = null; state.selectedRwCnt = null;
      renderAll();
    }""")
    page.wait_for_timeout(800)
    page.click("#listBody tr")
    page.wait_for_timeout(1200)

    def click_listeners():
        return page.evaluate("""() => {
          const gd = document.getElementById('chartDiv');
          const ev = gd && gd._ev && gd._ev._events;
          const h = ev && ev['plotly_click'];
          return !h ? 0 : (Array.isArray(h) ? h.length : 1);
        }""")

    assert click_listeners() == 1, "처음부터 안 걸려 있으면 이 검사는 뜻이 없다"
    page.evaluate("() => Plotly.purge('chartDiv')")
    assert click_listeners() == 0, "purge 가 발신기를 지우는 게 이 결함의 전제다"
    page.evaluate("() => renderAll()")
    page.wait_for_timeout(1200)
    assert click_listeners() == 1, "다시 그린 뒤에도 클릭이 죽어 있습니다"

    page.evaluate("""() => {
      state.page = 'wac';
      document.getElementById('pageDcOcap').hidden = true;
      document.getElementById('pageWac').hidden = false;
    }""")


def test_hover_width_counts_escaped_characters(wac_page):
    """hover 는 HTML 이라 "<" 는 "&lt;" 네 칸으로 그려진다.

    원문 기준으로 재면 부등호가 섞인 코멘트에서 상자가 기준보다 넓어져
    가리키던 타점을 덮는다.
    """
    page, _planted, _traces, _errors = wac_page
    got = page.evaluate("""() => ({
      plain: clipToWidth('abcdefghij', 6),
      angle: clipToWidth('<<<<<<', 9),
      width: drawnWidth('a<b'),
    })""")
    assert got["width"] == 1 + 4 + 1
    assert got["plain"] == "abcde…"
    # "<" 하나가 4 칸이므로 8칸(=2개)까지만 들어가고 "…" 자리가 남는다
    assert got["angle"] == "<<…"


def test_typing_in_the_search_box_rebuilds_the_grid_once(wac_page):
    """renderWac 은 그려둔 차트를 전부 purge 하고 그리드를 새로 만든다.

    글자마다 부르면 "item" 네 글자에 네 번 벌어진다. 한 박자 모았다가
    한 번만 그려야 한다.
    """
    page, _planted, _traces, _errors = wac_page
    page.fill("#wacSearch", "")
    page.wait_for_timeout(1200)
    page.evaluate("""() => {
      window.__wacCalls = 0;
      const orig = renderWac;
      renderWac = (...a) => { window.__wacCalls += 1; return orig(...a); };
      window.__restoreWac = () => { renderWac = orig; };
    }""")
    page.type("#wacSearch", "item", delay=40)
    page.wait_for_timeout(1500)
    calls = page.evaluate("() => window.__wacCalls")
    page.evaluate("() => window.__restoreWac()")
    assert page.eval_on_selector_all(".wac-chart", "d => d.length") > 0, (
        "검색 결과가 없으면 이 검사는 뜻이 없다")
    assert calls == 1, f"네 글자를 쳤는데 그리드를 {calls}번 새로 만들었습니다"


def test_typing_in_the_search_box_leaves_the_stat_charts_alone(wac_page):
    """통계 칸은 제품에만 달려 있다. 검색어가 바뀌었다고 다시 그릴 이유가 없다.

    다시 그리면 plotly 차트 네 개를 지웠다 만드는 일이 검색할 때마다 붙는다.
    """
    page, _planted, _traces, _errors = wac_page
    page.fill("#wacSearch", "")
    page.wait_for_timeout(1200)
    # purge -> react 를 거치면 svg 노드가 새로 생긴다. 그대로면 안 건드린 것이다.
    page.evaluate("() => { window.__svg = document.querySelector('#wacStatMonth svg'); }")
    assert page.evaluate("() => !!window.__svg"), "월별 통계 차트가 없습니다"
    page.type("#wacSearch", "item", delay=40)
    page.wait_for_timeout(1500)
    same = page.evaluate("() => document.querySelector('#wacStatMonth svg') === window.__svg")
    assert same, "검색만 바뀌었는데 통계 차트를 다시 그렸습니다"

    # 제품이 바뀌면 반드시 다시 그려야 한다 -- 가드가 너무 세면 옛 제품
    # 통계가 그대로 남는다
    page.evaluate("() => { state.product = 'TTS'; renderWac(); }")
    page.wait_for_timeout(1500)
    assert not page.evaluate(
        "() => document.querySelector('#wacStatMonth svg') === window.__svg"), (
        "제품을 바꿨는데 통계가 그대로입니다")


# ------------------------------------------------ 행당 chart 수

def _grid_info(page):
    return page.evaluate("""() => {
      const g = document.getElementById('wacGrid');
      const plot = document.querySelector('.wac-chart .js-plotly-plot');
      const doc = document.documentElement;
      return {
        cols: getComputedStyle(g).gridTemplateColumns.split(' ').length,
        pick: document.getElementById('wacPerRow').value,
        svg: plot ? Math.round(plot.querySelector('.main-svg').getBoundingClientRect().width) : null,
        overflow: doc.scrollWidth > doc.clientWidth,
      };
    }""")


def test_charts_per_row_defaults_to_two(wac_page):
    page, _planted, _traces, _errors = wac_page
    opts = page.evaluate(
        "() => [...document.getElementById('wacPerRow').options].map(o => Number(o.value))")
    assert opts == [2, 3, 4, 5]
    info = _grid_info(page)
    assert info["cols"] == 2 and info["pick"] == "2"


@pytest.mark.parametrize("n", [3, 4, 5])
def test_charts_per_row_actually_resizes_the_drawn_charts(wac_page, n):
    """격자 열 수만 바꾸면 plotly 는 옛 너비 그대로 남는다.

    plotly 의 responsive 는 window 크기 변화만 듣기 때문이다. 그러면 칸은
    좁아졌는데 그림은 안 좁아져서, 차트가 칸 밖으로 삐져나오거나 옆 차트를
    덮는다 -- 그래서 바꿀 때마다 크기가 바뀌었다고 알려줘야 한다.
    """
    page, _planted, _traces, _errors = wac_page
    before = _grid_info(page)
    page.select_option("#wacPerRow", str(n))
    page.wait_for_timeout(1600)
    after = _grid_info(page)
    assert after["cols"] == n, after
    assert after["svg"] < before["svg"], (
        f"열을 {n}개로 늘렸는데 그림 너비가 {before['svg']} 그대로입니다")
    assert not after["overflow"], "차트가 칸을 넘어 가로 스크롤이 생겼습니다"


def test_changing_charts_per_row_keeps_the_selection(wac_page):
    """열 수만 바뀐 것이라 골라둔 wafer 가 풀리면 안 된다."""
    page, planted, _traces, _errors = wac_page
    page.evaluate("""() => {
      const r = wacRows(state.product, DATA.itemCols[state.product][0])[0];
      state.wacSelected = { rootLotId: r.root_lot_id, waferId: r.wafer_id };
      wacApplySelection();
    }""")
    page.wait_for_timeout(600)
    before = page.evaluate("() => JSON.stringify(state.wacSelected)")
    page.select_option("#wacPerRow", "4")
    page.wait_for_timeout(1600)
    assert page.evaluate("() => JSON.stringify(state.wacSelected)") == before
    # 고른 타점은 커진 채로 남아 있어야 한다 (덧그림이 살아 있다는 뜻)
    assert page.evaluate(f"""() => wacCharts.filter(e => e.drawn).some(e => {{
      const ov = e.div.data[e.ovSel];
      return ov && ov.x.length > 0 && ov.marker.size.some(s => s === {app.WAC_SIZE_SEL});
    }})""")


def test_a_malformed_hold_time_does_not_invent_a_month(wac_page):
    """달은 hold_time 앞 7글자다. 길이만 보면 "2026-8-" 도 통과해서,
    월별 통계에 있지도 않은 달이 하나 더 생긴다."""
    page, _planted, _traces, _errors = wac_page
    months = page.evaluate("""() => {
      const orig = DATA.dc.ULY;
      DATA.dc.ULY = [
        { ...orig[0], lot_id: 'A', hold_time: '2026-08-30T01:00:00' },
        { ...orig[0], lot_id: 'B', hold_time: '2026-8-3 01:00:00' },
        { ...orig[0], lot_id: 'C', hold_time: '' },
      ];
      try { return wacStats('ULY').months; } finally { DATA.dc.ULY = orig; }
    }""")
    assert months == ["2026-08"], months


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


# ---------------------------------------------------- DC OCAP 목록 필터

def _filter(page, rows, view):
    """페이지의 filterByStatus 를 직접 불러 (lot_id, rw_cnt) 목록을 돌려준다."""
    return page.evaluate(
        """([rows, view]) => filterByStatus(rows, view)
             .map(r => r.lot_id + '|' + normRwCnt(r.rw_cnt))
             .filter((v, i, a) => a.indexOf(v) === i)""", [rows, view])


def _row(lot, rw, **over):
    r = {"lot_id": lot, "rw_cnt": rw, "root_lot_id": lot.split(".")[0], "wafer_id": 1,
         "item_id": "item1", "code": None, "owner": None, "comment": None, "status": "Hold"}
    r.update(over)
    return r


def test_a_reworked_lot_moves_its_earlier_hold_to_history(wac_page):
    """같은 lot 이 다시 걸리면 앞 건은 끝난 것이다 -- hold 에 남기지 않는다."""
    page, _planted, _traces, _errors = wac_page
    rows = [_row("A.1", 0), _row("A.1", 1)]
    assert _filter(page, rows, "hold") == ["A.1|1"]
    assert _filter(page, rows, "이력") == ["A.1|0"]


def test_supersession_follows_the_whole_rework_chain(wac_page):
    page, _planted, _traces, _errors = wac_page
    rows = [_row("A.1", 0), _row("A.1", 1), _row("A.1", 2)]
    assert _filter(page, rows, "hold") == ["A.1|2"]
    assert sorted(_filter(page, rows, "이력")) == ["A.1|0", "A.1|1"]


def test_supersession_is_scoped_to_one_lot(wac_page):
    """다른 lot 의 rw_cnt 가 높다고 이 lot 이 밀려나면 안 된다."""
    page, _planted, _traces, _errors = wac_page
    rows = [_row("A.1", 0), _row("B.1", 3)]
    assert sorted(_filter(page, rows, "hold")) == ["A.1|0", "B.1|3"]


def test_a_row_with_no_rw_cnt_is_neither_pushed_out_nor_pushes(wac_page):
    """순서를 매길 수 없는 줄은 건드리지 않는다."""
    page, _planted, _traces, _errors = wac_page
    rows = [_row("A.1", None), _row("A.1", 2)]
    assert sorted(_filter(page, rows, "hold")) == ["A.1|", "A.1|2"]


def test_a_dispositioned_rework_leaves_the_earlier_one_in_history(wac_page):
    """새 건이 이미 조치됐어도 앞 건이 hold 로 돌아오지는 않는다."""
    page, _planted, _traces, _errors = wac_page
    rows = [_row("A.1", 0), _row("A.1", 1, code="Flow", owner="김", status="Run")]
    assert _filter(page, rows, "hold") == []
    assert sorted(_filter(page, rows, "이력")) == ["A.1|0", "A.1|1"]


def test_one_hold_event_never_lands_in_both_lists(wac_page):
    """한 건 안에서 item 별로 조치가 갈려도 목록에는 한 번만 나와야 한다.

    목록은 (lot_id, rw_cnt) 마다 한 줄이라, 줄 단위로 가르면 같은 건이
    hold 와 이력 양쪽에 다 뜬다 (이 규칙을 넣기 전에 실제로 그랬다).
    """
    page, _planted, _traces, _errors = wac_page
    rows = [_row("A.1", 0, item_id="item1"),
            _row("A.1", 0, item_id="item2", code="Flow", owner="김", status="Run")]
    hold, hist = _filter(page, rows, "hold"), _filter(page, rows, "이력")
    assert set(hold) & set(hist) == set(), (hold, hist)
    assert len(hold) + len(hist) == 1


def test_the_whole_list_view_is_untouched(wac_page):
    page, _planted, _traces, _errors = wac_page
    rows = [_row("A.1", 0), _row("A.1", 1), _row("B.1", 0, code="Flow", owner="김")]
    assert len(_filter(page, rows, "전체")) == 3


@pytest.mark.parametrize("view", ["hold", "이력", "전체"])
def test_python_and_the_browser_split_the_list_the_same_way(wac_page, view):
    """같은 규칙이 두 파일에 손으로 적혀 있다 -- 실제로 한쪽만 고쳐서 갈렸다.

    (승계 규칙이 템플릿에만 들어가는 바람에, diagnose.py 가 알려주는
    hold 건수가 화면에 보이는 것보다 많았다.)
    """
    page, _planted, _traces, _errors = wac_page
    frames = app.frames_by_product(app.pull_data())
    for product in ("ULY", "TTS", "SOL"):
        dc = frames[0][product]
        rows = json.loads(dc.to_json(orient="records", date_format="iso"))
        js = sorted(_filter(page, rows, view))
        py = app.filter_by_status(dc, view)
        want = sorted({f"{r.lot_id}|{app.norm_rw_cnt(r.rw_cnt)}"
                       for r in py.itertuples()})
        assert js == want, f"{product}/{view}: 브라우저와 파이썬이 다릅니다"


# --------------------------------------------- 타점 선택 (진짜 마우스로)
#
# 여기는 g.emit(...) 같은 합성 이벤트를 쓰지 않는다. 실제 결함이 DOM click 과
# plotly_click 의 순서/간격에서 났고, 합성 이벤트는 그 경로를 안 지나서
# 아무것도 못 잡았다.

def _mouse_click_point(page, chart=0, nth=0):
    pos = page.evaluate(
        """([c, n]) => {
          const g = document.querySelectorAll('.wac-chart div.js-plotly-plot')[c];
          const ps = g.querySelectorAll('.points path');
          if (!ps[n]) return null;
          const r = ps[n].getBoundingClientRect();
          return [r.left + r.width / 2, r.top + r.height / 2];
        }""", [chart, nth])
    assert pos, "타점을 못 찾았습니다"
    page.mouse.click(pos[0], pos[1])
    page.wait_for_timeout(1200)


def _selected(page):
    return page.evaluate("() => state.wacSelected")


def _close_modal(page):
    page.evaluate("() => closeWacModal()")
    page.wait_for_timeout(200)


def test_clicking_the_same_point_again_clears_the_selection(wac_page):
    """조치 내역이 있는 타점도 다시 누르면 풀려야 한다.

    팝업을 띄우면서 선택을 덮어쓰기만 하면 그 wafer 는 영영 해제할 수 없다.
    """
    page, _planted, _traces, _errors = wac_page
    _mouse_click_point(page)
    assert _selected(page) is not None
    _close_modal(page)
    _mouse_click_point(page)
    assert _selected(page) is None, "다시 눌렀는데 안 풀렸습니다"


def test_clicking_empty_chart_space_clears_the_selection(wac_page):
    """어느 차트든 빈 곳을 누르면 선택이 풀린다."""
    page, _planted, _traces, _errors = wac_page
    _mouse_click_point(page)
    assert _selected(page) is not None
    _close_modal(page)
    empty = page.evaluate(
        """() => { const g = document.querySelectorAll('.wac-chart div.js-plotly-plot')[1];
                   const r = g.getBoundingClientRect(); return [r.left + 16, r.top + 12]; }""")
    page.mouse.click(empty[0], empty[1])
    page.wait_for_timeout(900)
    assert _selected(page) is None, "빈 곳을 눌렀는데 안 풀렸습니다"


def test_a_real_click_on_a_point_does_not_clear_what_it_just_selected(wac_page):
    """빈 곳 판정이 시간차였을 때, 타점 클릭이 스스로 지운 적이 있다.

    선택을 반영하느라 차트를 전부 다시 칠하는 데 0.6초가 걸려서, DOM click 이
    도착할 즈음엔 '방금 타점을 눌렀다' 는 시간 창이 이미 지나 있었다.
    """
    page, _planted, _traces, _errors = wac_page
    _mouse_click_point(page)
    assert _selected(page) is not None, "타점을 눌렀는데 선택이 비어 있습니다"


def test_clicking_the_legend_keeps_the_selection(wac_page):
    """legend 클릭은 그 차트를 다루려는 것이지 선택을 풀려는 게 아니다."""
    page, _planted, _traces, _errors = wac_page
    _mouse_click_point(page)
    _close_modal(page)
    before = _selected(page)
    box = page.evaluate(
        """() => { const l = document.querySelector('.wac-chart div.js-plotly-plot .legend .traces');
                   if (!l) return null; const r = l.getBoundingClientRect();
                   return [r.left + 8, r.top + 8]; }""")
    if not box:
        pytest.skip("legend 가 없는 차트다")
    page.mouse.click(box[0], box[1])
    page.wait_for_timeout(900)
    assert _selected(page) == before, "legend 를 눌렀는데 선택이 풀렸습니다"

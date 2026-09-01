"""데이터가 왜 안 보이는지 단계별로 찍어보는 진단 스크립트.

app.py 와 같은 폴더에 두고 실행:

    python diagnose.py

pull_data() 가 돌려준 데이터를 대시보드가 쓰는 순서 그대로 따라가면서,
어느 단계에서 건수가 0 이 되는지 / 어디서 예외가 나는지 보여줍니다.
화면에 아무것도 안 뜨거나 이상하게 뜰 때 제일 먼저 돌려볼 것.

한 단계가 예외로 죽어도 나머지는 계속 돌아갑니다. 맨 끝 '요약' 에
발견된 문제가 심각한 순서대로 다시 모여서 나옵니다.
"""
import traceback

import pandas as pd

import app


# 발견된 문제를 모아뒀다가 맨 끝 요약에서 한 번에 보여준다.
# (level: 1=치명적, 2=화면이 비어보임, 3=참고)
FINDINGS = []

# 같은 예외가 제품 3개 x 상태 3개로 아홉 번씩 터지면 traceback 이 화면을
# 도배해서 정작 진단 결과가 안 보인다. 처음 한 번만 전문을 찍고 그 다음
# 부터는 몇 번째인지만 알린다.
_SEEN_EXC = {}


def note(level, title, *lines):
    FINDINGS.append((level, title, [str(x) for x in lines]))


def report_exc(step_name, e, header=True):
    """예외를 한 번만 자세히, 나머지는 짧게 보고한다.

    traceback 도 print 로 내보낸다. stderr 로 보내면 파이프로 넘길 때
    나머지 출력과 순서가 섞여서 어느 단계에서 난 에러인지 알 수 없다.
    header=False 는 호출한 쪽이 이미 한 줄 찍은 경우(같은 내용을 두 번
    보여주지 않기 위해).
    """
    sig = (type(e).__name__, str(e))
    _SEEN_EXC[sig] = _SEEN_EXC.get(sig, 0) + 1
    n = _SEEN_EXC[sig]
    if n > 1:
        print(f"        (같은 에러 {n}번째, traceback 생략)")
        return
    if header:
        print(f"  !! {step_name}: 예외 발생 -> {type(e).__name__}: {e}")
    for line in traceback.format_exc().rstrip().splitlines():
        print("        " + line)
    note(1, f"{step_name} 실행 중 예외",
         f"{type(e).__name__}: {e}",
         "위 traceback 의 마지막 줄이 실제 원인입니다.")


def head(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def safe(step_name, fn, *args, **kwargs):
    """한 단계를 실행하되, 예외가 나면 traceback 을 찍고 계속 진행한다.

    진단 도중 죽어버리면 정작 원인이 있는 뒤쪽 단계를 못 보게 되므로,
    예외 자체도 '발견된 문제' 로 기록해두고 넘어간다.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        report_exc(step_name, e)
        return None


def blank_mask(df, col):
    """해당 컬럼이 비어있는(NaN 또는 공백) 행 마스크."""
    if col not in df.columns:
        return None
    return df[col].isna() | (df[col].astype(str).str.strip() == "")


def sample_lots(df, mask, n=3):
    """문제가 있는 행에서 lot_id 예시를 뽑는다 (사내 조회용)."""
    if "lot_id" not in df.columns:
        return []
    return list(dict.fromkeys(df.loc[mask, "lot_id"].astype(str)))[:n]


# ====================================================================
# 1~3. 형태 / 컬럼 / check_data
# ====================================================================
def check_shapes(product_dc, product_trend):
    for name, df in product_dc.items():
        if not isinstance(df, pd.DataFrame):
            print(f"  {name.lower()}_dc  : !! DataFrame 이 아님 ({type(df).__name__})")
            note(1, f"{name.lower()}_dc 가 DataFrame 이 아님",
                 f"실제 타입: {type(df).__name__}",
                 "pull_data() 가 DataFrame 을 돌려주도록 고쳐야 합니다.")
        else:
            print(f"  {name.lower()}_dc  : {len(df):>7,} 행 x {len(df.columns)} 컬럼"
                  + ("   << 비어 있음!" if df.empty else ""))
    for name, df in product_trend.items():
        if not isinstance(df, pd.DataFrame):
            print(f"  {name.lower()}_trend: !! DataFrame 이 아님 ({type(df).__name__})")
            note(1, f"{name.lower()}_trend 가 DataFrame 이 아님",
                 f"실제 타입: {type(df).__name__}")
        else:
            print(f"  {name.lower()}_trend: {len(df):>7,} 행 x {len(df.columns)} 컬럼"
                  + ("   << 비어 있음!" if df.empty else ""))

    if all(isinstance(d, pd.DataFrame) and d.empty for d in product_dc.values()):
        print("\n>> dc 3개가 모두 비어 있습니다. 대시보드에 아무것도 안 뜨는 것이 당연합니다.")
        print("   pull_data() 안의 조회 조건(기간/제품/라인 등)을 확인하세요.")
        note(1, "dc 3개가 전부 비어 있음",
             "pull_data() 안의 조회 조건(기간/제품/라인)을 확인하세요.")


def check_columns(product_dc, product_trend):
    for name, df in product_dc.items():
        if not isinstance(df, pd.DataFrame):
            continue
        missing = [c for c in app.DC_REQUIRED if c not in df.columns]
        print(f"  {name.lower()}_dc: " + ("전부 있음" if not missing else f"!! 없음 -> {missing}"))
        if missing:
            print(f"     실제 컬럼: {list(df.columns)}")
            note(1, f"{name.lower()}_dc 에 필수 컬럼이 없음",
                 f"없는 컬럼: {missing}",
                 "이 상태면 대시보드가 아예 에러 화면을 띄우고 멈춥니다.")
    for name, df in product_trend.items():
        if not isinstance(df, pd.DataFrame):
            continue
        missing = [c for c in app.TREND_REQUIRED if c not in df.columns]
        print(f"  {name.lower()}_trend: " + ("전부 있음" if not missing else f"!! 없음 -> {missing}"))
        if missing:
            print(f"     실제 컬럼: {list(df.columns)[:15]}{' ...' if len(df.columns) > 15 else ''}")
            note(1, f"{name.lower()}_trend 에 필수 컬럼이 없음", f"없는 컬럼: {missing}")

    # status 는 DC_REQUIRED 에 없다 (없어도 돌아가도록 되어 있음).
    # 다만 없으면 hold 필터가 예전처럼 owner/code 만 보고 판단한다.
    for name, df in product_dc.items():
        if isinstance(df, pd.DataFrame) and "status" not in df.columns:
            print(f"  {name.lower()}_dc: (참고) status 컬럼이 없습니다 "
                  "-> hold 판정에 status 조건이 빠집니다")
            note(3, f"{name.lower()}_dc 에 status 컬럼이 없음",
                 "owner/comment 가 다음날 아침에 적재되는 문제를 status 로 막고 있는데,",
                 "이 제품은 그 조건이 빠진 채로 동작합니다.")


def check_spec(product_spec, product_trend):
    """관리선 개정 이력. 없으면 그 item 은 회색으로만 그려진다."""
    for name, df in product_spec.items():
        if not isinstance(df, pd.DataFrame):
            print(f"  {name.lower()}_spec: !! DataFrame 이 아님 ({type(df).__name__})")
            note(1, f"{name.lower()}_spec 가 DataFrame 이 아님")
            continue
        missing = [c for c in app.SPEC_REQUIRED if c not in df.columns]
        line = f"  {name.lower()}_spec: {len(df):>7,} 행"
        if missing:
            print(line + f"   !! 컬럼 없음 -> {missing}")
            note(1, f"{name.lower()}_spec 에 필수 컬럼이 없음", f"없는 컬럼: {missing}")
            continue
        if df.empty:
            print(line + "   << 비어 있음!")
            note(2, f"{name.lower()}_spec 가 비어 있음",
                 "CL OUT / SL OUT 판정이 안 되고 전부 회색으로 그려집니다.")
            continue
        # 개정이 여러 번인 item 이 있으면 그 차트에 계단이 생긴다
        per_item = df.groupby("item_id", observed=True).size()
        stepped = int((per_item > 1).sum())
        print(line + f"  (item {len(per_item)}종, 규격이 바뀐 item {stepped}종)")

        tr = product_trend.get(name)
        if isinstance(tr, pd.DataFrame) and not tr.empty:
            have = set(df["item_id"].astype(str).str.strip().str.lower())
            items = [str(c) for c in app.item_columns(tr)]
            miss = [i for i in items if i.strip().lower() not in have]
            if miss:
                print(f"          !! spec 에 없는 item {len(miss)}/{len(items)}종: {miss[:5]}")
                note(1 if len(miss) == len(items) else 2,
                     f"[{name}] spec 에 없는 item {len(miss)}/{len(items)}종",
                     f"예: {miss[:5]}",
                     "해당 차트는 관리선 없이 회색으로만 그려집니다.")


def check_split(product_split, product_trend):
    """EIN/ECN 적용 이력. 아직 화면에 쓰지는 않지만, 한 테이블을 process_id
    로 잘라 만드는 것이라 '잘못 잘라서 통째로 빈' 경우가 제일 흔하다."""
    expected = {name: cfg["process_id"] for name, cfg in app.PRODUCT_CONFIG.items()}
    wafer_cols = [str(n) for n in range(1, 26)]
    for name, df in product_split.items():
        if not isinstance(df, pd.DataFrame):
            print(f"  {name.lower()}_split: !! DataFrame 이 아님 ({type(df).__name__})")
            note(2, f"{name.lower()}_split 가 DataFrame 이 아님")
            continue
        line = f"  {name.lower()}_split: {len(df):>7,} 행"
        if df.empty:
            print(line + f"   << 비어 있음! (process_id == '{expected[name]}' 로 잘랐는지 확인)")
            note(2, f"{name.lower()}_split 가 비어 있음",
                 f"split 을 process_id == '{expected[name]}' 로 자르는 부분을 확인하세요.")
            continue
        # 자를 때 제품을 바꿔 넣으면 여기서 바로 드러난다
        got = sorted(str(v) for v in df.get("process_id", pd.Series(dtype=str)).unique())
        if got != [expected[name]]:
            print(line + f"   !! process_id 가 {got} 입니다 ('{expected[name]}' 여야 함)")
            note(1, f"{name.lower()}_split 의 process_id 가 다름",
                 f"들어온 값: {got} / 기대값: {expected[name]}",
                 "split 을 자르는 순서나 조건이 제품과 어긋났습니다.")
            continue
        missing = [c for c in wafer_cols if c not in df.columns]
        if missing:
            print(line + f"   !! wafer 칸이 없음 -> {missing[:5]}{' ...' if len(missing) > 5 else ''}")
            note(2, f"{name.lower()}_split 에 wafer 칸(1~25)이 없음",
                 f"없는 칸: {missing[:5]}",
                 "comp_id_list 를 1~25 칸으로 펼치는 단계가 빠졌는지 확인하세요.")
            continue
        marks = (df[wafer_cols] == "V").sum(axis=1)
        blank = int((marks == 0).sum())
        print(line + f"  (V 없는 행 {blank}건, 행마다 V {int(marks.min())}~{int(marks.max())}개)")
        if blank:
            note(3, f"[{name}] split 에 V 가 하나도 없는 행 {blank}건",
                 "wafer 를 하나도 안 건드린 건이라, 보통은 앞 단계에서 버립니다.")
        tr = product_trend.get(name)
        if isinstance(tr, pd.DataFrame) and not tr.empty and "root_lot_id" in df.columns:
            unknown = set(df["root_lot_id"]) - set(tr["root_lot_id"])
            if unknown:
                print(f"          (참고) trend 에 없는 lot {len(unknown)}개: "
                      f"{sorted(map(str, unknown))[:3]}")
                note(3, f"[{name}] split 의 lot 중 trend 에 없는 것 {len(unknown)}개",
                     "조회 기간이 서로 다르면 정상입니다.")


def check_check_data(product_dc, product_trend, product_spec):
    result = safe("check_data()", app.check_data, product_dc, product_trend, product_spec)
    if result is None:
        return
    problems, warnings = result
    if warnings:
        # 대시보드는 정상 동작하는 항목들. 화면을 막지 않으므로 참고로만.
        print(f"  [참고 {len(warnings)}건 - 대시보드는 정상 동작]")
        for w in warnings:
            print("    · " + w)
        note(3, "check_data() 참고사항 (화면은 정상 동작)", *warnings)
    if problems:
        for p in problems:
            print("  - " + p)
        note(1, "check_data() 가 문제를 보고했습니다",
             *problems,
             "이 상태면 대시보드가 에러 화면을 띄우고 멈춥니다.")
    else:
        print("  문제 없음")


# ====================================================================
# 4. 핵심 컬럼 값 품질 (hold_time / rw_cnt / status)
# ====================================================================
def check_key_columns(product_dc):
    """리스트가 '있긴 한데 이상하게' 나오는 원인을 잡는 단계.

    hold_time / rw_cnt / status 세 개는 리스트를 어떻게 쪼개고 무엇을
    hold 로 볼지를 결정한다. 컬럼이 있어도 값이 틀어져 있으면 화면은
    에러 없이 '조용히' 틀리게 나오므로 여기서 따로 본다.
    """
    for name, df in product_dc.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"\n  [{name}] dc 가 비어 있음")
            continue
        print(f"\n  [{name}]")

        # ---- hold_time ---------------------------------------------
        if "hold_time" in df.columns:
            dtype = df["hold_time"].dtype
            is_dt = pd.api.types.is_datetime64_any_dtype(df["hold_time"])
            nat = int(df["hold_time"].isna().sum())
            print(f"     hold_time  dtype={dtype}  결측={nat:,}/{len(df):,}"
                  + ("" if is_dt else "   << 날짜형이 아닙니다"))
            if not is_dt:
                note(2, f"[{name}] hold_time 이 날짜형이 아님",
                     f"현재 dtype={dtype}",
                     "문자열이면 정렬이 사전순이 되어 최신순 정렬이 틀어지고,",
                     "rw_cnt 를 hold_time 순위로 매기는 계산도 같이 틀어집니다.",
                     "pull_data() 에서 pd.to_datetime() 으로 변환해 주세요.")
            if nat:
                note(2, f"[{name}] hold_time 이 비어있는 행 {nat:,}건",
                     f"예시 lot_id: {sample_lots(df, df['hold_time'].isna())}",
                     "hold_time 이 비면 rw_cnt 순위 계산이 NaN 이 됩니다.")

        # ---- rw_cnt -------------------------------------------------
        if "rw_cnt" in df.columns:
            na = int(df["rw_cnt"].isna().sum())
            print(f"     rw_cnt     dtype={df['rw_cnt'].dtype}  결측={na:,}/{len(df):,}"
                  f"  값 종류={sorted(set(df['rw_cnt'].dropna().map(app.norm_rw_cnt)))[:8]}")
            if na:
                note(2, f"[{name}] rw_cnt 가 비어있는 행 {na:,}건",
                     f"예시 lot_id: {sample_lots(df, df['rw_cnt'].isna())}",
                     "화면은 안 깨지지만(빈 값도 한 묶음으로 처리), rw_cnt 가 비었다는 건",
                     "보통 hold_time 이 비어서 순위 계산이 실패했다는 신호입니다.")

            if "lot_id" in df.columns and "hold_time" in df.columns:
                # rw_cnt 는 'hold 이벤트' 단위여야 한다. wafer 측정 단위로
                # 들어오면 같은 hold 가 여러 줄로 쪼개져 신규 건수가 부풀고,
                # 반대로 재작업인데 같은 번호면 원본과 한 줄로 합쳐진다.
                g = df.groupby(["lot_id", "hold_time"], dropna=False, observed=True)["rw_cnt"]
                split = g.nunique(dropna=False)
                split = split[split > 1]
                if len(split):
                    print(f"     !! 같은 (lot_id, hold_time) 인데 rw_cnt 가 여러 개: {len(split)}건")
                    print(f"        예: {[f'{a} / {b}' for a, b in list(split.index)[:3]]}")
                    note(1, f"[{name}] 한 hold 이벤트가 여러 줄로 쪼개집니다",
                         f"같은 (lot_id, hold_time) 인데 rw_cnt 가 다른 경우: {len(split)}건",
                         f"예: {[f'{a} / {b}' for a, b in list(split.index)[:3]]}",
                         "rw_cnt 는 wafer 측정 단위가 아니라 hold 이벤트 단위여야 합니다.",
                         "lot_id 안에서 hold_time 을 dense rank 한 값인지 확인하세요.",
                         "이 상태면 '신규 hold 건수' 가 실제보다 부풀어 보입니다.")

                g2 = df.groupby(["lot_id", "rw_cnt"], dropna=False, observed=True)["hold_time"]
                merged = g2.nunique(dropna=False)
                merged = merged[merged > 1]
                if len(merged):
                    print(f"     !! 같은 (lot_id, rw_cnt) 인데 hold_time 이 여러 개: {len(merged)}건")
                    print(f"        예: {[f'{a} / rw_cnt={b}' for a, b in list(merged.index)[:3]]}")
                    note(1, f"[{name}] 재작업 건이 원본과 한 줄로 합쳐집니다",
                         f"같은 (lot_id, rw_cnt) 인데 hold_time 이 다른 경우: {len(merged)}건",
                         f"예: {[f'{a} / rw_cnt={b}' for a, b in list(merged.index)[:3]]}",
                         "재측정 후 다시 hold 가 걸렸으면 rw_cnt 가 올라가야 하는데",
                         "그대로라서 리스트에서 원본과 한 줄로 묶입니다.")

        # ---- status -------------------------------------------------
        if "status" in df.columns:
            txt = df["status"].astype(str).str.strip()
            blank = df["status"].isna() | txt.isin(["", "nan", "None", "NaT"])
            counts = txt[~blank].value_counts()
            print(f"     status     결측={int(blank.sum()):,}/{len(df):,}  "
                  f"값: {dict(list(counts.items())[:6])}")

            # hold 판정은 'Hold' (H 만 대문자) 를 정확히 본다.
            wrong_case = [v for v in counts.index
                          if v.lower() == "hold" and v != "Hold"]
            if wrong_case:
                n = int(counts[wrong_case].sum())
                print(f"     !! 'Hold' 와 대소문자가 다른 값: {wrong_case} ({n:,}행)")
                note(1, f"[{name}] status 대소문자가 달라 hold 가 전부 이력으로 빠집니다",
                     f"발견된 표기: {wrong_case} ({n:,}행)",
                     "hold 판정은 정확히 'Hold' (H 만 대문자) 만 인정합니다.",
                     "원본 테이블 표기를 확인하거나 pull_data() 에서 맞춰주세요.")
            # 'Hold' 가 아니면 빈 값도 전부 이력이다. 그래서 status 결측은
            # 그 자체로 'hold 에서 사라진 건수' 다. code/owner 까지 비어
            # 있는(= 원래 hold 로 보여야 할) 행만 세어야 과장이 안 된다.
            cb, ob = blank_mask(df, "code"), blank_mask(df, "owner")
            if cb is not None and ob is not None:
                hidden = blank & cb & ob
                if int(hidden.sum()):
                    print(f"     !! status 가 비어서 hold 에서 빠지는 행: {int(hidden.sum()):,}")
                    note(1, f"[{name}] status 미적재로 hold 에서 빠지는 행 {int(hidden.sum()):,}건",
                         f"예시 lot_id: {sample_lots(df, hidden)}",
                         "code/owner 가 둘 다 비었으니 원래는 hold 로 보여야 할 행인데,",
                         "status 가 'Hold' 가 아니라(비어 있어서) 이력으로 넘어갑니다.",
                         "status 조인이 신규 hold 를 못 따라오고 있는지 확인하세요.")
            if not len(counts):
                note(1, f"[{name}] status 값이 전부 비어 있음",
                     "'Hold' 가 아닌 것은 전부 이력이므로 hold 탭이 통째로 0건이 됩니다.",
                     "status 테이블 조인이 실패한 건 아닌지 확인하세요.")


# ====================================================================
# 5. 상태 필터별 건수
# ====================================================================
def check_status_filter(product_dc):
    print("   (대시보드 기본값은 'hold' 입니다. hold 가 0 이면 첫 화면이 빈 채로 보입니다)")
    for name, df in product_dc.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"\n  [{name}] dc 가 비어 있음")
            continue
        print(f"\n  [{name}]  전체 {len(df):,} 행")
        # 각 탭이 '언제부터 언제까지' 를 담고 있는지 같이 찍는다. "이력에
        # 과거 게 안 보인다" 같은 신고가 들어왔을 때, 필터가 걸러낸 건지
        # 애초에 dc 에 안 들어온 건지를 이 한 줄로 가른다.
        for view in ("전체", "hold", "이력"):
            try:
                sub = app.filter_by_status(df, view)
                lots = app.group_holds(sub)
                mark = "   << 0건" if len(lots) == 0 else ""
                span = ""
                if "hold_time" in sub.columns and len(sub) and sub["hold_time"].notna().any():
                    lo, hi = sub["hold_time"].min(), sub["hold_time"].max()
                    span = f"  기간 {str(lo)[:10]} ~ {str(hi)[:10]}"
                print(f"     {view:4s}: {len(sub):>7,} 행 -> 리스트 {len(lots):>4,} 줄{span}{mark}")
            except Exception as e:
                print(f"     {view:4s}: !! {type(e).__name__}: {e}")
                report_exc(f"[{name}] '{view}' 필터", e, header=False)

        # '전체' 가 담고 있는 기간 자체가 짧으면 필터 문제가 아니라
        # pull_data() 가 과거를 안 가져온 것이다. status 컬럼을 붙일 때
        # inner join 을 쓰면 status 테이블에 없는 과거 lot 이 통째로
        # 빠지면서 딱 이 모양이 된다.
        if "hold_time" in df.columns and df["hold_time"].notna().any():
            lo, hi = df["hold_time"].min(), df["hold_time"].max()
            try:
                days = (hi - lo).days
            except Exception:
                days = None
            if days is not None and days <= 7:
                note(2, f"[{name}] dc 가 담고 있는 기간이 {days}일뿐입니다",
                     f"{str(lo)[:16]} ~ {str(hi)[:16]}",
                     "'이력' 에 과거가 안 보인다면 필터가 아니라 pull_data() 문제입니다.",
                     "status 컬럼을 merge(how='inner') 로 붙이면 status 테이블에 없는",
                     "과거 lot 이 통째로 빠집니다. how='left' 로 바꾸세요.")

        cb, ob = blank_mask(df, "code"), blank_mask(df, "owner")
        if cb is not None and ob is not None:
            print(f"     code 비어있는 행 : {int(cb.sum()):,} / {len(df):,}")
            print(f"     owner 비어있는 행: {int(ob.sum()):,} / {len(df):,}")
            if int((cb & ob).sum()) == 0:
                print("     >> code/owner 가 둘 다 빈 행이 하나도 없습니다.")
                print("        = 'hold' 필터는 항상 0건입니다. 화면에서 '전체' 를 눌러보세요.")
                note(2, f"[{name}] code/owner 가 둘 다 빈 행이 없음",
                     "'hold' 필터가 항상 0건이 되어 첫 화면이 비어 보입니다.")


# ====================================================================
# 6. dc <-> trend 연결
# ====================================================================
def check_join(product_dc, product_trend):
    for name in product_dc:
        dc_df, tr_df = product_dc[name], product_trend.get(name)
        if not isinstance(dc_df, pd.DataFrame) or not isinstance(tr_df, pd.DataFrame):
            continue
        if dc_df.empty or tr_df.empty:
            print(f"  [{name}] dc 또는 trend 가 비어 있어 확인 불가")
            continue
        print(f"\n  [{name}]")

        if "item_id" in dc_df.columns:
            ids = list(dict.fromkeys(dc_df["item_id"]))
            unresolved = [i for i in ids if app.resolve_item_col(tr_df, i) is None]
            print(f"     item_id 종류 {len(ids)}개 중 trend 컬럼과 매칭 안 되는 것: "
                  f"{len(unresolved)}개")
            if unresolved:
                print(f"       예: {unresolved[:5]}")
                print(f"       trend 쪽 컬럼 예: {list(tr_df.columns)[:10]}")
                note(2, f"[{name}] item_id 가 trend 컬럼명과 매칭되지 않음",
                     f"매칭 안 되는 item: {unresolved[:5]}",
                     f"trend 쪽 컬럼 예: {list(tr_df.columns)[:10]}",
                     "해당 item 을 고르면 차트 자리에 경고만 뜹니다.")

        pair_cols = {"root_lot_id", "wafer_id"}
        if pair_cols <= set(dc_df.columns) and pair_cols <= set(tr_df.columns):
            dc_pairs = set(zip(dc_df["root_lot_id"].map(app.norm_lot),
                               dc_df["wafer_id"].map(app.norm_wafer)))
            tr_pairs = set(zip(tr_df["root_lot_id"].map(app.norm_lot),
                               tr_df["wafer_id"].map(app.norm_wafer)))
            miss = dc_pairs - tr_pairs
            print(f"     (root_lot_id, wafer_id) {len(dc_pairs)}쌍 중 "
                  f"trend 에 없는 것: {len(miss)}쌍")
            if miss:
                print(f"       예(dc): {sorted(map(str, miss))[:3]}")
                print(f"       예(trend): {sorted(map(str, tr_pairs))[:3]}")
                note(2, f"[{name}] dc 의 wafer 가 trend 에 없습니다",
                     f"{len(miss)}/{len(dc_pairs)} 쌍이 매칭 실패",
                     f"예(dc): {sorted(map(str, miss))[:3]}",
                     f"예(trend): {sorted(map(str, tr_pairs))[:3]}",
                     "차트에 빨간/파란 점(=hold 걸린 wafer)이 안 찍힙니다.",
                     "양쪽 lot_id 표기(접두어/자리수)나 wafer_id 형식을 맞춰야 합니다.")
            print(f"     dtype  dc.wafer_id={dc_df['wafer_id'].dtype} "
                  f"trend.wafer_id={tr_df['wafer_id'].dtype}")


# ====================================================================
# 7. 관리선(spec) 이 시간순으로 말이 되는가
# ====================================================================
def _limit_tuple(row):
    """한 행의 관리선 4개를 비교 가능한 튜플로. 숫자가 아니면 None."""
    out = []
    for w in app.LIMIT_COLS:
        v = app.to_float(row.get(w))
        out.append(None if v is None else round(v, 6))
    return tuple(out)


def check_spec_quality(product_spec, product_trend):
    """spec 을 '개정 이력' 으로 읽었을 때 앞뒤가 맞는지 본다.

    spec 은 (item_id, from_time) 마다 한 벌이어야 하고, 시간이 갈수록
    한 방향으로 바뀌어야 합니다. 실제로 자주 깨지는 건 두 가지입니다:

      · 같은 item + 같은 from_time 에 규격이 두 벌 -- 그러면 merge_asof
        가 그 중 아무거나(뒤에 오는 행) 집어가고, 차트의 관리선이
        조회할 때마다 달라 보입니다.
      · A -> B -> A 처럼 값이 되돌아옴 -- 이건 개정이 아니라 서로 다른
        기준 두 벌이 한 테이블에 섞여 들어온 것입니다. item_id 와
        시각만으로는 어느 쪽인지 못 가리므로, 뽑는 쿼리에 조건을 하나
        더 걸어야 합니다 (설비/스텝/라인 등).

    둘 다 화면에는 에러가 안 뜨고 '관리선이 왔다갔다' 로만 보입니다.
    """
    for name, df in product_spec.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if [c for c in app.SPEC_REQUIRED if c not in df.columns]:
            continue
        sp = df[app.SPEC_REQUIRED].copy()
        sp["from_time"] = pd.to_datetime(sp["from_time"], errors="coerce")
        n_bad_time = int(sp["from_time"].isna().sum())
        sp = sp.dropna(subset=["from_time"])
        sp["_key"] = sp["item_id"].astype(str).str.strip().str.lower()
        print(f"\n  [{name}]  {len(df):,} 행 / item {sp['_key'].nunique()} 종")
        if n_bad_time:
            print(f"     !! from_time 이 날짜로 안 읽히는 행 {n_bad_time}건 -> 그 개정은 통째로 빠짐")
            note(2, f"[{name}] spec.from_time 이 날짜로 안 읽히는 행 {n_bad_time}건",
                 "그 개정은 관리선 계산에서 통째로 빠집니다.")

        # 관리선이 4개 다 비어있는 행: 규격 설정 전의 껍데기 행
        empty_rows = int(sp.apply(lambda r: _limit_tuple(r) == (None,) * 4, axis=1).sum())
        if empty_rows:
            print(f"     !! 관리선이 4개 다 비어있는 행 {empty_rows}/{len(sp)}건")
            note(2, f"[{name}] spec 에 관리선이 전부 빈 행 {empty_rows}건",
                 "규격 설정 전 행으로 보입니다. 뽑을 때 걸러내세요:",
                 "  et_data = et_data[pd.to_numeric(et_data['ucl'], errors='coerce').notna()]")

        dup_items, flip_items, inv_items = [], [], []
        for key, grp in sp.groupby("_key", observed=True, sort=False):
            grp = grp.sort_values("from_time")
            tuples = [_limit_tuple(r) for _, r in grp.iterrows()]
            times = list(grp["from_time"])

            # (1) 같은 시각에 서로 다른 규격
            by_time = {}
            for t, tup in zip(times, tuples):
                by_time.setdefault(t, set()).add(tup)
            clash = [t for t, s in by_time.items() if len(s) > 1]
            if clash:
                dup_items.append((key, len(clash), min(clash)))

            # (2) 값이 되돌아옴 (A -> B -> A)
            seq = [t for i, t in enumerate(tuples) if i == 0 or t != tuples[i - 1]]
            if len(seq) > len(set(seq)):
                flip_items.append((key, len(set(seq)), len(grp)))

            # (3) 규격 자체가 뒤집힘
            for t, (ucl, lcl, usl, lsl) in zip(times, tuples):
                if None in (usl, lsl) or usl > lsl:
                    if None in (ucl, lcl) or ucl > lcl:
                        continue
                inv_items.append((key, t))
                break

        if dup_items:
            worst = sorted(dup_items, key=lambda x: -x[1])[:5]
            print(f"     !! 같은 item + 같은 from_time 에 규격이 2벌 이상: {len(dup_items)} 종")
            for k, n, t in worst:
                print(f"        {k}: {n}개 시각에서 충돌 (예: {t})")
            note(1, f"[{name}] 같은 item+시각에 규격이 2벌인 item {len(dup_items)}종",
                 *[f"{k}: {n}개 시각 충돌 (예: {t})" for k, n, t in worst],
                 "merge_asof 가 그 중 뒤 행을 집어가므로 관리선이 임의로 정해집니다.",
                 "item_id + from_time 이 유일해지도록 한 벌만 남기고 뽑으세요.")
        if flip_items:
            worst = sorted(flip_items, key=lambda x: -x[2])[:5]
            print(f"     !! 값이 되돌아오는(A->B->A) item: {len(flip_items)} 종")
            for k, distinct, rows in worst:
                print(f"        {k}: {rows}행인데 규격은 {distinct}벌뿐 -- 두 기준이 섞여 있음")
            note(2, f"[{name}] 규격이 A->B->A 로 되돌아오는 item {len(flip_items)}종",
                 *[f"{k}: {rows}행 / 규격 {distinct}벌" for k, distinct, rows in worst],
                 "정상 개정이면 값은 한 방향으로만 갑니다. 되돌아온다는 건",
                 "서로 다른 기준 두 벌이 한 테이블에 섞였다는 뜻입니다.",
                 "item_id 말고 어떤 컬럼이 두 벌을 가르는지 찾아서 조건을 거세요",
                 "(그 컬럼으로 나눴을 때 각 그룹의 규격이 한 벌이면 그게 답입니다).")
        if inv_items:
            print(f"     !! 상한 <= 하한 인 item: {len(inv_items)} 종 (예: {inv_items[0][0]})")
            note(2, f"[{name}] 상한이 하한보다 작거나 같은 item {len(inv_items)}종",
                 *[f"{k} ({t})" for k, t in inv_items[:5]],
                 "usl/lsl 또는 ucl/lcl 이 서로 바뀐 것 같습니다.")

        # 개정 시점이 trend 구간과 맞물리는가
        late_start = 0
        tr = product_trend.get(name)
        if isinstance(tr, pd.DataFrame) and not tr.empty and "tkout_time" in tr.columns:
            t_min, t_max = tr["tkout_time"].min(), tr["tkout_time"].max()
            later = int((sp["from_time"] > t_max).sum())
            first = sp.groupby("_key", observed=True)["from_time"].min()
            late_start = int((first > t_min).sum())
            print(f"     trend 구간 {t_min:%Y-%m-%d} ~ {t_max:%Y-%m-%d} 기준: "
                  f"구간 뒤 개정 {later}행, 첫 개정이 구간 시작보다 늦은 item {late_start}종")
            if late_start:
                note(3, f"[{name}] 첫 개정이 trend 시작보다 늦은 item {late_start}종",
                     "그 이전 구간은 관리선이 없어 회색으로만 그려집니다 "
                     "(spec 을 trend 보다 넉넉히 뽑으면 사라집니다).")
        if not (dup_items or flip_items or inv_items or empty_rows
                or n_bad_time or late_start):
            print("     이상 없음")


# ====================================================================
# 8. 리스트 -> 우측 패널 시뮬레이션
# ====================================================================
def check_row_click(product_dc, product_trend):
    """리스트의 각 줄을 실제로 눌러본 것처럼 우측 패널을 계산해 본다.

    리스트에는 멀쩡히 떠 있는데 클릭하면 차트/코멘트가 통째로 비는
    경우가 있다. 리스트는 (lot_id, rw_cnt) 로 묶고 우측은 그 두 개로
    되짚어 찾는 구조라, 그 되짚기가 실패하면 화면에는 아무 에러도 안
    뜨고 그냥 빈 칸이 된다. 여기서 미리 잡는다.
    """
    for name, df in product_dc.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        tr_df = product_trend.get(name)
        print(f"\n  [{name}]")

        for view in ("hold", "전체"):
            dc_df = safe(f"[{name}] filter_by_status('{view}')",
                         app.filter_by_status, df, view)
            if dc_df is None:
                continue
            grouped = safe(f"[{name}] group_holds('{view}')", app.group_holds, dc_df)
            if grouped is None or grouped.empty:
                print(f"     {view:4s}: 리스트가 비어 있어 확인 생략")
                continue

            empty_rows, no_chart = [], []
            for _, row in grouped.iterrows():
                lot_id, rw_cnt = row["lot_id"], row.get("rw_cnt", "")
                lot_rows = dc_df[
                    (dc_df["lot_id"] == lot_id)
                    & dc_df["rw_cnt"].map(app.norm_rw_cnt).eq(app.norm_rw_cnt(rw_cnt))
                ] if "rw_cnt" in dc_df.columns else dc_df[dc_df["lot_id"] == lot_id]

                items = list(dict.fromkeys(lot_rows["item_id"])) if len(lot_rows) else []
                if not items:
                    empty_rows.append(f"{lot_id} (rw_cnt={rw_cnt!r})")
                    continue
                if isinstance(tr_df, pd.DataFrame) and not tr_df.empty:
                    if app.resolve_item_col(tr_df, items[0]) is None:
                        no_chart.append(f"{lot_id} (rw_cnt={rw_cnt}, item={items[0]})")

            ok = len(grouped) - len(empty_rows)
            print(f"     {view:4s}: {len(grouped)}줄 중 {ok}줄 정상"
                  + (f", {len(empty_rows)}줄 우측 공백" if empty_rows else "")
                  + (f", {len(no_chart)}줄 차트 없음" if no_chart else ""))

            if empty_rows:
                print(f"        우측 공백 예: {empty_rows[:3]}")
                note(1, f"[{name}/{view}] 리스트에는 뜨는데 클릭하면 우측이 빕니다",
                     f"{len(empty_rows)}/{len(grouped)} 줄",
                     f"예: {empty_rows[:3]}",
                     "리스트 줄을 자기 원본 행으로 되짚지 못한 경우입니다.",
                     "위 4번의 rw_cnt / hold_time 항목을 먼저 확인하세요.")
            if no_chart:
                print(f"        차트 없음 예: {no_chart[:3]}")
                note(3, f"[{name}/{view}] 첫 item 의 trend 컬럼을 못 찾습니다",
                     f"{len(no_chart)}줄", f"예: {no_chart[:3]}")


# ====================================================================
# 9. 요약
# ====================================================================
def summary(product_dc):
    total_lots = hold_lots = 0
    for df in product_dc.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            try:
                total_lots += len(app.group_holds(app.filter_by_status(df, "전체")))
                hold_lots += len(app.group_holds(app.filter_by_status(df, "hold")))
            except Exception:
                pass
    print(f"  전체 기준 리스트 줄 수 : {total_lots:,}")
    print(f"  hold 기준 리스트 줄 수 : {hold_lots:,}  (대시보드 첫 화면에 보이는 것)")
    print()
    if not total_lots:
        print("  >> 어느 상태로도 표시할 행이 없습니다. 리스트가 통째로 빕니다.")
    elif not hold_lots:
        print("  >> 리스트에 표시할 데이터는 있는데 hold 가 0건입니다.")
        print("     대시보드 기본 필터가 'hold' 라서 첫 화면만 비어 보이는 것입니다.")
        print("     화면에서 '전체' 를 눌러보세요.")
    else:
        print("  >> 리스트 자체는 정상입니다 (첫 화면에 hold 건이 보여야 정상).")

    if not FINDINGS:
        print("\n  >> 발견된 문제 없음.")
        return

    label = {1: "[치명적]", 2: "[화면 이상]", 3: "[참고]"}
    print(f"\n  발견된 문제 {len(FINDINGS)}건 (심각한 순):")
    for level, title, lines in sorted(FINDINGS, key=lambda f: f[0]):
        print(f"\n  {label[level]} {title}")
        for line in lines:
            print(f"      {line}")


def main():
    head("1. pull_data() 가 무엇을 돌려줬나")
    try:
        frames = app.pull_data()
    except Exception:
        print("!! pull_data() 자체가 실패했습니다:")
        for line in traceback.format_exc().rstrip().splitlines():
            print("   " + line)
        print("\n>> 여기서 막히면 뒤 단계는 볼 것도 없습니다.")
        print("   위 traceback 의 마지막 줄이 실제 원인입니다 "
              "(조회 권한 / 쿼리 문법 / 접속 정보 등).")
        return

    if not isinstance(frames, (tuple, list)) or len(frames) != 12:
        n = len(frames) if hasattr(frames, "__len__") else "?"
        print(f"!! 12개를 return 해야 하는데 {type(frames).__name__} ({n}개) 를 돌려줬습니다.")
        print("   순서: uly_dc, sol_dc, tts_dc, uly_trend, sol_trend, tts_trend,")
        print("         uly_spec, sol_spec, tts_spec, uly_split, sol_split, tts_split")
        print("   spec_* 는 (item_id, from_time, ucl, lcl, usl, lsl) 개정 이력입니다.")
        print("   split_* 는 EIN/ECN 적용 이력입니다 (split 을 process_id 로 나눈 것).")
        return

    (uly_dc, sol_dc, tts_dc, uly_trend, sol_trend, tts_trend,
     uly_spec, sol_spec, tts_spec, uly_split, sol_split, tts_split) = frames
    product_dc = {"ULY": uly_dc, "SOL": sol_dc, "TTS": tts_dc}
    product_trend = {"ULY": uly_trend, "SOL": sol_trend, "TTS": tts_trend}
    product_spec = {"ULY": uly_spec, "SOL": sol_spec, "TTS": tts_spec}
    product_split = {"ULY": uly_split, "SOL": sol_split, "TTS": tts_split}
    safe("1단계", check_shapes, product_dc, product_trend)
    safe("1단계(spec)", check_spec, product_spec, product_trend)
    safe("1단계(split)", check_split, product_split, product_trend)

    head("2. 컬럼 이름 확인 (대시보드가 요구하는 것 대비)")
    safe("2단계", check_columns, product_dc, product_trend)

    head("3. check_data() 결과 (이게 걸리면 화면에 에러가 떴어야 함)")
    safe("3단계", check_check_data, product_dc, product_trend, product_spec)

    head("4. hold_time / rw_cnt / status 값 확인  << 조용히 틀리는 원인")
    safe("4단계", check_key_columns, product_dc)

    head("5. 상태 필터별 건수  << 화면이 비어 보이는 가장 흔한 원인")
    safe("5단계", check_status_filter, product_dc)

    head("6. dc 와 trend 가 서로 연결되는가 (차트가 비는 원인)")
    safe("6단계", check_join, product_dc, product_trend)

    head("7. 관리선(spec) 이 시간순으로 말이 되는가  << 관리선이 왔다갔다 할 때")
    safe("7단계", check_spec_quality, product_spec, product_trend)

    head("8. 리스트 행을 클릭하면 우측이 뜨는가 (시뮬레이션)")
    safe("8단계", check_row_click, product_dc, product_trend)

    head("9. 요약")
    safe("9단계", summary, product_dc)


if __name__ == "__main__":
    main()

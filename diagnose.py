"""데이터가 왜 안 보이는지 단계별로 찍어보는 진단 스크립트.

app.py 와 같은 폴더에 두고 실행:

    python diagnose.py

pull_data() 가 돌려준 데이터를 대시보드가 쓰는 순서 그대로 따라가면서,
어느 단계에서 건수가 0 이 되는지 보여줍니다. 화면에 아무것도 안 뜰 때
제일 먼저 돌려볼 것.
"""
import sys
import traceback

import pandas as pd

import app


def head(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def blank_mask(df, col):
    """해당 컬럼이 비어있는(NaN 또는 공백) 행 마스크."""
    if col not in df.columns:
        return None
    return df[col].isna() | (df[col].astype(str).str.strip() == "")


def main():
    head("1. pull_data() 가 무엇을 돌려줬나")
    try:
        frames = app.pull_data()
    except Exception:
        print("!! pull_data() 자체가 실패했습니다:")
        traceback.print_exc()
        return

    if not isinstance(frames, (tuple, list)) or len(frames) != 6:
        print(f"!! 6개를 return 해야 하는데 {type(frames).__name__} "
              f"({len(frames) if hasattr(frames, '__len__') else '?'}개) 를 돌려줬습니다.")
        print("   순서: dc_uly, dc_sol, dc_tts, uly_trend, sol_trend, tts_trend")
        return

    dc_uly, dc_sol, dc_tts, uly_trend, sol_trend, tts_trend = frames
    product_dc = {"ULY": dc_uly, "SOL": dc_sol, "TTS": dc_tts}
    product_trend = {"ULY": uly_trend, "SOL": sol_trend, "TTS": tts_trend}

    for name, df in list(product_dc.items()):
        if not isinstance(df, pd.DataFrame):
            print(f"  dc_{name.lower():4s}: !! DataFrame 이 아님 ({type(df).__name__})")
        else:
            print(f"  dc_{name.lower():4s}: {len(df):>7,} 행 x {len(df.columns)} 컬럼"
                  + ("   << 비어 있음!" if df.empty else ""))
    for name, df in list(product_trend.items()):
        if not isinstance(df, pd.DataFrame):
            print(f"  {name.lower()}_trend: !! DataFrame 이 아님 ({type(df).__name__})")
        else:
            print(f"  {name.lower()}_trend: {len(df):>7,} 행 x {len(df.columns)} 컬럼"
                  + ("   << 비어 있음!" if df.empty else ""))

    if all(isinstance(d, pd.DataFrame) and d.empty for d in product_dc.values()):
        print("\n>> dc 3개가 모두 비어 있습니다. 대시보드에 아무것도 안 뜨는 것이 당연합니다.")
        print("   pull_data() 안의 조회 조건(기간/제품/라인 등)을 확인하세요.")

    head("2. 컬럼 이름 확인 (대시보드가 요구하는 것 대비)")
    for name, df in product_dc.items():
        if not isinstance(df, pd.DataFrame):
            continue
        missing = [c for c in app.DC_REQUIRED if c not in df.columns]
        print(f"  dc_{name.lower()}: " + ("전부 있음" if not missing else f"!! 없음 -> {missing}"))
        if missing:
            print(f"     실제 컬럼: {list(df.columns)}")
    for name, df in product_trend.items():
        if not isinstance(df, pd.DataFrame):
            continue
        missing = [c for c in app.TREND_REQUIRED if c not in df.columns]
        print(f"  {name.lower()}_trend: " + ("전부 있음" if not missing else f"!! 없음 -> {missing}"))
        if missing:
            print(f"     실제 컬럼: {list(df.columns)[:15]}{' ...' if len(df.columns) > 15 else ''}")

    head("3. check_data() 결과 (이게 걸리면 화면에 에러가 떴어야 함)")
    try:
        problems = app.check_data(product_dc, product_trend)
    except Exception:
        print("!! check_data() 가 예외로 죽었습니다:")
        traceback.print_exc()
        problems = None
    if problems is None:
        pass
    elif problems:
        for p in problems:
            print("  - " + p)
    else:
        print("  문제 없음")

    head("4. 상태 필터별 건수  << 화면이 비어 보이는 가장 흔한 원인")
    print("   (대시보드 기본값은 'hold' 입니다. hold 가 0 이면 첫 화면이 빈 채로 보입니다)")
    for name, df in product_dc.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"\n  [{name}] dc 가 비어 있음")
            continue
        print(f"\n  [{name}]  전체 {len(df):,} 행")
        for status in ("전체", "hold", "이력"):
            try:
                sub = app.filter_by_status(df, status)
                lots = app.group_holds(sub)
                mark = "   << 0건" if len(lots) == 0 else ""
                print(f"     {status:4s}: {len(sub):>7,} 행 -> 리스트 {len(lots):>4,} 줄{mark}")
            except Exception as e:
                print(f"     {status:4s}: !! {type(e).__name__}: {e}")

        cb, ob = blank_mask(df, "code"), blank_mask(df, "owner")
        if cb is not None and ob is not None:
            print(f"     code 비어있는 행 : {int(cb.sum()):,} / {len(df):,}")
            print(f"     owner 비어있는 행: {int(ob.sum()):,} / {len(df):,}")
            if int((cb & ob).sum()) == 0:
                print("     >> code/owner 가 둘 다 빈 행이 하나도 없습니다.")
                print("        = 'hold' 필터는 항상 0건입니다. 화면에서 '전체' 를 눌러보세요.")

    head("5. dc 와 trend 가 서로 연결되는가 (차트가 비는 원인)")
    broken = {"item": [], "pair": []}
    for name in ("ULY", "SOL", "TTS"):
        dc_df, tr_df = product_dc[name], product_trend[name]
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
                broken["item"].append(name)
                print(f"       예: {unresolved[:5]}")
                print(f"       trend 쪽 컬럼 예: {[c for c in tr_df.columns][:10]}")

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
                broken["pair"].append((name, len(miss), len(dc_pairs)))
                print(f"       예(dc): {sorted(map(str, miss))[:3]}")
                print(f"       예(trend): {sorted(map(str, tr_pairs))[:3]}")
            print(f"     dtype  dc.wafer_id={dc_df['wafer_id'].dtype} "
                  f"trend.wafer_id={tr_df['wafer_id'].dtype}")

    head("6. 요약")
    total_lots = 0
    hold_lots = 0
    for name, df in product_dc.items():
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
        print("     위 1~3번(pull_data 결과 / 컬럼 이름 / check_data)을 확인하세요.")
    elif not hold_lots:
        print("  >> 리스트에 표시할 데이터는 있는데 hold 가 0건입니다.")
        print("     대시보드 기본 필터가 'hold' 라서 첫 화면만 비어 보이는 것입니다.")
        print("     화면에서 '전체' 를 눌러보세요.")
    else:
        print("  >> 리스트 자체는 정상입니다 (첫 화면에 hold 건이 보여야 정상).")

    if broken["pair"]:
        print()
        print("  >> 리스트에서 행을 골라도 차트에 빨간/파란 점이 안 찍힙니다:")
        for name, miss, tot in broken["pair"]:
            print(f"     [{name}] dc 의 (root_lot_id, wafer_id) {miss}/{tot} 쌍이"
                  f" trend 에 없습니다.")
        print("     양쪽 lot_id 표기(접두어/자리수)나 wafer_id 형식을 맞춰야 합니다.")
        print("     바로 위 5번에 양쪽 실제 값 예시가 찍혀 있으니 비교해보세요.")
    if broken["item"]:
        print()
        print(f"  >> [{', '.join(broken['item'])}] item_id 가 trend 컬럼명과 매칭되지"
              " 않습니다. 해당 item 은 차트가 안 뜹니다.")


if __name__ == "__main__":
    main()

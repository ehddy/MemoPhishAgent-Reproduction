"""
MemoPhishAgent 평가 스크립트

Usage:
    # 단일 클래스 모드
    python evaluate.py \
        --phish  results/openphish_result.json \
        --benign results/tranco_result.json

    # 혼합 모드 (ground_truth 컬럼 포함 JSON)
    python evaluate.py --mixed results/mixed_result.json

입력 형식 (test_dataset.py 출력 JSON):
    {
      "results": [{"url": ..., "malicious": bool, "ground_truth": "phish"|"benign", ...}, ...],
      "run_info": {"total_urls": ..., "llm_calls": ..., "token_usage": {...}, ...}
    }

중복 제거는 test_dataset.py 단계에서 이미 완료된다.
"""

import argparse
import json
import os


# ---------------------------------------------------------------------------
# 로딩
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def records_from_phish_benign(phish_path: str, benign_path: str) -> tuple[list, list]:
    """단일 클래스 JSON 두 개를 읽어 records 리스트와 run_info 리스트를 반환한다."""
    phish_data  = load_json(phish_path)
    benign_data = load_json(benign_path)

    phish_records = [
        {"gt": True, "pred": bool(r.get("malicious", False))}
        for r in phish_data.get("results", [])
    ]
    benign_records = [
        {"gt": False, "pred": bool(r.get("malicious", False))}
        for r in benign_data.get("results", [])
    ]
    run_infos = [
        ri for ri in [phish_data.get("run_info", {}), benign_data.get("run_info", {})]
        if ri
    ]
    return phish_records + benign_records, run_infos


def records_from_mixed(mixed_path: str) -> tuple[list, list]:
    """혼합 JSON(ground_truth 컬럼 포함)을 읽어 records 리스트와 run_info 리스트를 반환한다."""
    data = load_json(mixed_path)
    records = []
    for r in data.get("results", []):
        gt_raw = r.get("ground_truth", "")
        gt = (str(gt_raw).strip().lower() == "phish")
        pred = bool(r.get("malicious", False))
        records.append({"gt": gt, "pred": pred})
    run_infos = [data["run_info"]] if data.get("run_info") else []
    return records, run_infos


# ---------------------------------------------------------------------------
# 통계 집계
# ---------------------------------------------------------------------------

def merge_run_infos(infos: list[dict]) -> dict:
    """여러 run_info를 합산해 통합 통계를 만든다."""
    if not infos:
        return {}

    merged: dict = {
        "total_urls": 0,
        "completed": 0,
        "failed": 0,
        "total_elapsed_sec": 0.0,
        "llm_calls": 0,
        "token_usage": {},
    }
    for ri in infos:
        merged["total_urls"]        += ri.get("total_urls", 0)
        merged["completed"]         += ri.get("completed", 0)
        merged["failed"]            += ri.get("failed", 0)
        merged["total_elapsed_sec"] += ri.get("total_elapsed_sec", 0.0)
        merged["llm_calls"]         += ri.get("llm_calls", 0)

        for model, usage in ri.get("token_usage", {}).items():
            if model not in merged["token_usage"]:
                merged["token_usage"][model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "input_token_details": {"cache_read": 0, "cache_creation": 0},
                }
            t = merged["token_usage"][model]
            t["input_tokens"]  += usage.get("input_tokens", 0)
            t["output_tokens"] += usage.get("output_tokens", 0)
            t["total_tokens"]  += usage.get("total_tokens", 0)
            details = usage.get("input_token_details", {})
            t["input_token_details"]["cache_read"]     += details.get("cache_read", 0)
            t["input_token_details"]["cache_creation"] += details.get("cache_creation", 0)

    n = merged["total_urls"]
    merged["avg_elapsed_sec"] = merged["total_elapsed_sec"] / n if n else 0.0
    merged["avg_llm_calls"]   = merged["llm_calls"] / n if n else 0.0
    return merged


# ---------------------------------------------------------------------------
# 분류 지표
# ---------------------------------------------------------------------------

def compute_metrics(records: list[dict]) -> dict:
    TP = sum(1 for r in records if     r["gt"] and     r["pred"])
    TN = sum(1 for r in records if not r["gt"] and not r["pred"])
    FP = sum(1 for r in records if not r["gt"] and     r["pred"])
    FN = sum(1 for r in records if     r["gt"] and not r["pred"])

    total     = TP + TN + FP + FN
    accuracy  = (TP + TN) / total if total else 0
    precision = TP / (TP + FP)    if (TP + FP) else 0
    recall    = TP / (TP + FN)    if (TP + FN) else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0)

    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                accuracy=accuracy, precision=precision,
                recall=recall, f1=f1)


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------

def print_results(m: dict, phish_count: int, benign_count: int, run_info: dict) -> None:
    total_urls = phish_count + benign_count
    print("=" * 55)
    print("  MemoPhishAgent Evaluation Results")
    print("=" * 55)
    print(f"  Dataset  phish={phish_count}  benign={benign_count}  total={total_urls}")
    print()
    print("  [Confusion Matrix]")
    print(f"    TP={m['TP']}  FP={m['FP']}")
    print(f"    FN={m['FN']}  TN={m['TN']}")
    print()
    print("  [Classification Metrics]")
    print(f"    Accuracy  : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"    Precision : {m['precision']:.4f}")
    print(f"    Recall    : {m['recall']:.4f}")
    print(f"    F1 Score  : {m['f1']:.4f}")

    if not run_info:
        print("=" * 55)
        return

    n         = run_info.get("total_urls", 0)
    completed = run_info.get("completed", n)
    failed    = run_info.get("failed", 0)
    total_sec = run_info.get("total_elapsed_sec", 0.0)
    avg_sec   = run_info.get("avg_elapsed_sec", 0.0)
    llm_total = run_info.get("llm_calls", 0)
    avg_llm   = run_info.get("avg_llm_calls", llm_total / n if n else 0.0)

    print()
    print("  [Processing Stats]")
    print(f"    URLs total    : {n}  (completed={completed}, failed={failed})")
    print(f"    Elapsed total : {total_sec:.1f}s")
    print(f"    Elapsed avg   : {avg_sec:.2f}s / URL")

    print()
    print("  [LLM Call Stats]")
    print(f"    LLM calls total : {llm_total}")
    print(f"    LLM calls avg   : {avg_llm:.2f} / URL")

    token_usage = run_info.get("token_usage", {})
    if token_usage:
        print()
        print("  [Token Usage]")
        for model, usage in token_usage.items():
            model_short = model.replace("global.anthropic.", "")
            inp     = usage.get("input_tokens", 0)
            out     = usage.get("output_tokens", 0)
            total   = usage.get("total_tokens", 0)
            cache_r = usage.get("input_token_details", {}).get("cache_read", 0)
            cache_c = usage.get("input_token_details", {}).get("cache_creation", 0)
            avg_inp   = inp   / n if n else 0.0
            avg_out   = out   / n if n else 0.0
            avg_total = total / n if n else 0.0
            print(f"    Model : {model_short}")
            print(f"      Input  tokens : {inp:>10,}  (avg {avg_inp:>8.1f} / URL)")
            print(f"      Output tokens : {out:>10,}  (avg {avg_out:>8.1f} / URL)")
            print(f"      Total  tokens : {total:>10,}  (avg {avg_total:>8.1f} / URL)")
            if cache_r or cache_c:
                print(f"      Cache read    : {cache_r:>10,}")
                print(f"      Cache created : {cache_c:>10,}")

    print("=" * 55)


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MemoPhishAgent 결과 평가")
    parser.add_argument("--phish",  default=None, help="피싱 데이터셋 결과 JSON (ground truth=phish)")
    parser.add_argument("--benign", default=None, help="정상 데이터셋 결과 JSON (ground truth=benign)")
    parser.add_argument(
        "--mixed",
        default=None,
        help="혼합 모드 결과 JSON (ground_truth 필드 포함)",
    )
    args = parser.parse_args()

    if args.mixed:
        records, run_infos = records_from_mixed(args.mixed)
        phish_count  = sum(1 for r in records if r["gt"])
        benign_count = sum(1 for r in records if not r["gt"])
    elif args.phish and args.benign:
        records, run_infos = records_from_phish_benign(args.phish, args.benign)
        phish_count  = sum(1 for r in records if r["gt"])
        benign_count = sum(1 for r in records if not r["gt"])
    else:
        parser.error("--mixed 또는 (--phish + --benign) 중 하나를 지정하세요.")

    m        = compute_metrics(records)
    run_info = merge_run_infos(run_infos)
    print_results(m, phish_count, benign_count, run_info)


if __name__ == "__main__":
    main()
"""
MemoPhishAgent 평가 스크립트

Usage:
    python evaluate.py \
        --phish  results/sample_openphish.json \
        --benign results/sample_tranco.json
"""

import argparse
import json


def load_results(path: str, ground_truth: bool):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results  = data.get("results", [])
    run_info = data.get("run_info", {})

    records = [
        {"gt": ground_truth, "pred": bool(r.get("malicious", False))}
        for r in results
    ]
    return records, run_info


def compute_metrics(records):
    TP = sum(1 for r in records if r["gt"] and r["pred"])
    TN = sum(1 for r in records if not r["gt"] and not r["pred"])
    FP = sum(1 for r in records if not r["gt"] and r["pred"])
    FN = sum(1 for r in records if r["gt"] and not r["pred"])

    total     = TP + TN + FP + FN
    accuracy  = (TP + TN) / total if total else 0
    precision = TP / (TP + FP)    if (TP + FP) else 0
    recall    = TP / (TP + FN)    if (TP + FN) else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0)

    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                accuracy=accuracy, precision=precision,
                recall=recall, f1=f1)


def main():
    parser = argparse.ArgumentParser(description="MemoPhishAgent 결과 평가")
    parser.add_argument("--phish",  required=True, help="피싱 데이터셋 결과 JSON (ground truth=True)")
    parser.add_argument("--benign", required=True, help="정상 데이터셋 결과 JSON (ground truth=False)")
    args = parser.parse_args()

    phish_records,  phish_info  = load_results(args.phish,  ground_truth=True)
    benign_records, benign_info = load_results(args.benign, ground_truth=False)

    all_records = phish_records + benign_records
    m = compute_metrics(all_records)

    # 토큰/LLM 사용량 합산
    def get_tokens(info):
        usage = info.get("token_usage", {})
        return {
            "input":  usage.get("input_tokens",  0),
            "output": usage.get("output_tokens", 0),
            "total":  usage.get("total_tokens",  0),
        }

    pt = get_tokens(phish_info)
    bt = get_tokens(benign_info)

    total_input  = pt["input"]  + bt["input"]
    total_output = pt["output"] + bt["output"]
    total_tokens = pt["total"]  + bt["total"]
    total_llm    = phish_info.get("llm_calls", 0) + benign_info.get("llm_calls", 0)
    total_urls   = len(all_records)
    avg_tokens   = round(total_tokens / total_urls, 1) if total_urls else 0

    # ── 출력 ──────────────────────────────────────────
    print("=" * 50)
    print("  MemoPhishAgent Evaluation Results")
    print("=" * 50)
    print(f"  Dataset       phish={len(phish_records)}  benign={len(benign_records)}  total={total_urls}")
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
    print()
    print("  [Token & LLM Usage]")
    print(f"    Total tokens      : {total_tokens:,}  (in={total_input:,} / out={total_output:,})")
    print(f"    Avg tokens / URL  : {avg_tokens:,}")
    print(f"    Total LLM calls   : {total_llm}")
    print(f"    Avg LLM calls/URL : {round(total_llm / total_urls, 1) if total_urls else 0}")
    print("=" * 50)


if __name__ == "__main__":
    main()
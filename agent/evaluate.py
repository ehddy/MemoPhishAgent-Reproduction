"""
MemoPhishAgent 평가 스크립트

Usage:
    python evaluate.py \
        --phish  results/openphish_result.tsv \
        --benign results/tranco_result.tsv \
        [--phish-failed  results/openphish_result_failed_urls.txt] \
        [--benign-failed results/tranco_result_failed_urls.txt]

중복 제거 정책:
    - folder == "unknown" 행은 제외 (에이전트가 자체적으로 크롤한 서브 URL)
    - 동일 folder가 복수 행으로 나타날 경우 첫 번째 행만 사용

실패 URL 처리:
    - --phish-failed  제공 시: 실패한 피싱 URL을 FN으로 집계 (에이전트가 탐지 못함)
    - --benign-failed 제공 시: 실패한 정상 URL을 TN으로 집계 (피싱으로 분류하지 않음)
"""

import argparse
import csv


def load_tsv(path: str, ground_truth: bool):
    """TSV를 읽어 records 리스트와 집계된 URL 집합을 반환한다."""
    seen_folders = set()
    seen_urls = set()
    records = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            folder = row.get("folder", "").strip()
            if not folder or folder == "unknown":
                continue
            if folder in seen_folders:
                continue
            seen_folders.add(folder)
            url = row.get("url", "").strip()
            if url:
                seen_urls.add(url)
            pred = row.get("prediction", "").strip().lower() == "phish"
            records.append({"gt": ground_truth, "pred": pred})
    return records, seen_urls


def count_failed(path: str, already_counted: set) -> int:
    """_failed_urls.txt에서 실패 URL 수를 반환한다.
    TSV에서 이미 집계된 URL(already_counted)은 제외해 이중 집계를 방지한다."""
    if path is None:
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(
            1 for line in f
            if line.strip() and line.strip() not in already_counted
        )


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
    parser.add_argument("--phish",         required=True,  help="피싱 데이터셋 결과 TSV")
    parser.add_argument("--benign",        required=True,  help="정상 데이터셋 결과 TSV")
    parser.add_argument("--phish-failed",  default=None,   help="피싱 실패 URL 목록 (_failed_urls.txt)")
    parser.add_argument("--benign-failed", default=None,   help="정상 실패 URL 목록 (_failed_urls.txt)")
    args = parser.parse_args()

    phish_records,  phish_urls  = load_tsv(args.phish,  ground_truth=True)
    benign_records, benign_urls = load_tsv(args.benign, ground_truth=False)

    phish_failed  = count_failed(args.phish_failed,  phish_urls)
    benign_failed = count_failed(args.benign_failed, benign_urls)

    # 실패한 피싱 URL → FN (에이전트가 탐지 못함)
    phish_records  += [{"gt": True,  "pred": False}] * phish_failed
    # 실패한 정상 URL → TN (피싱으로 분류하지 않음)
    benign_records += [{"gt": False, "pred": False}] * benign_failed

    all_records = phish_records + benign_records
    m = compute_metrics(all_records)
    total_urls = len(all_records)

    print("=" * 50)
    print("  MemoPhishAgent Evaluation Results")
    print("=" * 50)
    print(f"  Dataset  phish={len(phish_records)}  benign={len(benign_records)}  total={total_urls}")
    if phish_failed or benign_failed:
        print(f"  (failed  phish={phish_failed}→FN  benign={benign_failed}→TN)")
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
    print("=" * 50)


if __name__ == "__main__":
    main()
"""
MemoPhishAgent 평가 스크립트

Usage:
    python evaluate.py \
        --phish  results/openphish_result.tsv \
        --benign results/tranco_result.tsv

중복 제거 정책:
    - folder == "unknown" 행은 제외 (에이전트가 자체적으로 크롤한 서브 URL)
    - 동일 folder가 복수 행으로 나타날 경우 첫 번째 행만 사용

실패 URL 처리:
    test_dataset.py가 실패 URL을 benign(prediction=benign)으로 TSV에 직접 기록하므로
    별도 _failed_urls.txt 입력 없이 항상 입력 수 = 결과 수가 보장된다.
"""

import argparse
import csv


def load_tsv(path: str, ground_truth: bool):
    seen_folders = set()
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
            pred = row.get("prediction", "").strip().lower() == "phish"
            records.append({"gt": ground_truth, "pred": pred})
    return records


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
    parser.add_argument("--phish",  required=True, help="피싱 데이터셋 결과 TSV (ground truth=phish)")
    parser.add_argument("--benign", required=True, help="정상 데이터셋 결과 TSV (ground truth=benign)")
    args = parser.parse_args()

    phish_records  = load_tsv(args.phish,  ground_truth=True)
    benign_records = load_tsv(args.benign, ground_truth=False)

    all_records = phish_records + benign_records
    m = compute_metrics(all_records)
    total_urls = len(all_records)

    print("=" * 50)
    print("  MemoPhishAgent Evaluation Results")
    print("=" * 50)
    print(f"  Dataset  phish={len(phish_records)}  benign={len(benign_records)}  total={total_urls}")
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

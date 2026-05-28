"""
MemoPhishAgent 평가 스크립트

Usage:
    # 단일 클래스 모드 (기존 방식)
    python evaluate.py \
        --phish  results/openphish_result.tsv \
        --benign results/tranco_result.tsv

    # 혼합 모드 (test_dataset.py --phish-root + --benign-root 결과)
    python evaluate.py --mixed results/mixed_result.tsv

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
    """단일 클래스 TSV 로드 (ground_truth 외부 지정)."""
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


def load_mixed_tsv(path: str):
    """혼합 모드 TSV 로드 (ground_truth 컬럼 포함).

    test_dataset.py --phish-root + --benign-root 출력 형식:
        folder  url  ground_truth  prediction  confidence  reason
    """
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
            gt   = row.get("ground_truth", "").strip().lower() == "phish"
            pred = row.get("prediction",   "").strip().lower() == "phish"
            records.append({"gt": gt, "pred": pred})
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


def print_results(m, phish_count, benign_count):
    total_urls = phish_count + benign_count
    print("=" * 50)
    print("  MemoPhishAgent Evaluation Results")
    print("=" * 50)
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
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="MemoPhishAgent 결과 평가")
    parser.add_argument("--phish",  default=None, help="피싱 데이터셋 결과 TSV (ground truth=phish)")
    parser.add_argument("--benign", default=None, help="정상 데이터셋 결과 TSV (ground truth=benign)")
    parser.add_argument(
        "--mixed",
        default=None,
        help="혼합 모드 결과 TSV (ground_truth 컬럼 포함, test_dataset.py --phish-root + --benign-root 출력)",
    )
    args = parser.parse_args()

    if args.mixed:
        records = load_mixed_tsv(args.mixed)
        phish_count  = sum(1 for r in records if r["gt"])
        benign_count = sum(1 for r in records if not r["gt"])
        m = compute_metrics(records)
        print_results(m, phish_count, benign_count)
    elif args.phish and args.benign:
        phish_records  = load_tsv(args.phish,  ground_truth=True)
        benign_records = load_tsv(args.benign, ground_truth=False)
        all_records = phish_records + benign_records
        m = compute_metrics(all_records)
        print_results(m, len(phish_records), len(benign_records))
    else:
        parser.error("--mixed 또는 (--phish + --benign) 중 하나를 지정하세요.")


if __name__ == "__main__":
    main()

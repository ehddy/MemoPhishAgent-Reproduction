# MemoPhishAgent — Reproduction

> **이 저장소는 아래 논문을 재현하기 위해 원본 [MemoPhishAgent](https://github.com/memophishagent/MemoPhishAgent)를 개인적으로 fork한 것입니다.
> 논문에서 제안하는 시스템을 직접 구현·실행해보며 결과를 검증하는 것이 목적입니다.**
>
> **This is a personal reproduction fork of the original [MemoPhishAgent](https://github.com/memophishagent/MemoPhishAgent) repository.
> The purpose is to reproduce the experiments described in the paper below, verify the reported results, and gain a deeper understanding of the system.**

## Paper

**Title:** MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection

**OpenReview:** [openreview.net/forum?id=xKHYnW2tsO](https://openreview.net/forum?id=xKHYnW2tsO)

## Original Repository

[github.com/memophishagent/MemoPhishAgent](https://github.com/memophishagent/MemoPhishAgent)

## Purpose of This Fork

This fork was created to **reproduce the experiments** described in the paper above.
The goal is to verify the reported results and gain a deeper understanding of the system.

Key differences from the original (environment-specific changes):

| 항목 | 원본 코드 | 이 재현본 |
| --- | --- | --- |
| LLM 클래스 | `ChatBedrock` | `ChatBedrock` (동일) |
| 모델 ID | `anthropic.claude-3-sonnet-20240229-v1:0` | `global.anthropic.claude-sonnet-4-6` (최신 모델로 교체) |
| AWS 리전 | `us-east-1` | `us-east-1` (동일) |
| LangChain 임포트 | `langchain.schema`, `langchain.tools` | `langchain_core.messages`, `langchain_core.tools` (v1.x 호환) |
| S3 FAISS 룩업 | 활성화 (저자 private 버킷) | 비활성화 (`None`) — 접근 불가 |
| 설정 파일 | `serpAPI_key.txt`, `test_urls.txt` 직접 커밋 | `.gitignore`로 제외, `.env.example` 템플릿 제공 |
| 오프라인 평가 | 없음 | `test_dataset.py` 추가 — 수집된 데이터셋(html.txt, shot.png)으로 live 접속 없이 배치 평가 |

> The original paper uses `anthropic.claude-3-sonnet-20240229-v1:0` which is a legacy model.
> This reproduction uses `global.anthropic.claude-sonnet-4-6` (a cross-region inference profile)
> to access the latest Claude model via AWS Bedrock.

---

## 🛡️ MemoPhishAgent: Multi-modal Agent-based End‑to‑End Phishing‑URL Detection Platform

MemoPhishAgent is a multi-modal agent-based end‑to‑end framework for collecting and detecting malicious or impersonation URLs at Internet scale.
It leverages memory-augmented reasoning via LangGraph ReAct agents backed by AWS Bedrock.

## 📂 Repository Structure

```text
.
├── agent/                  # Multi‑modal agent detection framework
│   ├── src/
│   │   ├── graph.py            # Live URL inference entry point
│   │   ├── test_dataset.py     # Offline batch evaluation (pre-collected datasets)
│   │   └── ...
│   ├── results/
│   │   └── samples/
│   │       ├── sample_openphish.tsv    # Sample phishing result (10 URLs, ground truth=phish)
│   │       └── sample_tranco.tsv       # Sample benign result (10 URLs, ground truth=benign)
│   ├── evaluate.py             # Offline evaluation metrics script
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example            # Environment variable template (AWS + SerpAPI)
│   └── test_urls.txt.example   # Input URL list template
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites

- Docker **or** Conda (Python 3.11)
- AWS credentials with Bedrock access (us-east-1)
- [SerpAPI](https://serpapi.com/) API key

### 2. Setup

```bash
# Clone this repo
git clone https://github.com/ehddy/MemoPhishAgent-Reproduction.git
cd MemoPhishAgent-Reproduction/agent

# Copy and fill in environment variables
cp .env.example .env
# Edit .env — fill in AWS credentials and SerpAPI key

cp test_urls.txt.example test_urls.txt
# Edit test_urls.txt and add URLs to analyze (one per line)
```

`.env` 파일 내용:

```dotenv
AWS_ACCESS_KEY_ID=your_access_key_id_here
AWS_SECRET_ACCESS_KEY=your_secret_access_key_here
AWS_REGION=us-east-1
SERPAPI_API_KEY=your_serpapi_key_here
```

### 3. Run with Docker — Live URL inference

```bash
docker build -t memophish-agent .

docker run --rm \
  --env-file .env \
  -v "$(pwd)/test_urls.txt":/work/test_urls.txt \
  -v "$(pwd)/results":/work/results \
  memophish-agent \
  python src/graph.py \
    --agent full_agent \
    --input test_urls.txt \
    --output results/result.json \
    --use-memory false \
    --use-ai-overview false
```

### 4. Run with Docker — Offline dataset evaluation

Pre-collected 데이터셋(html.txt, shot.png, info.txt)을 사용해 live 접속 없이 배치 평가합니다.
TR-OP 데이터셋처럼 이미 죽은 피싱 사이트 URL을 평가할 때 사용합니다.

```bash
docker run --rm \
  --env-file .env \
  -v "/path/to/datasets/openphish_5000":/datasets/openphish_5000:ro \
  -v "$(pwd)/results":/work/results \
  memophish-agent \
  python src/test_dataset.py \
    --dataset-root /datasets/openphish_5000 \
    --output results/openphish_result.json \
    --agent full_agent \
    --use-memory false \
    --use-ai-overview false
```

결과 파일:

| 파일 | 내용 |
| --- | --- |
| `results/openphish_result.json` | 판별 결과 JSON (`url`, `malicious`, `confidence`, `reason`) |
| `results/openphish_result.tsv` | TSV: `folder / url / prediction / confidence / reason` |
| `results/openphish_result_failed_urls.txt` | 처리 실패 URL 목록 |

### 5. Run locally with Conda

#### 5-1. Environment setup (one-time)

```bash
# Create and activate a Python 3.11 conda environment
conda create -n memophish python=3.11 -y
conda activate memophish

# Move into the agent directory
cd MemoPhishAgent-Reproduction/agent

# Install Python dependencies
pip install -r requirements.txt
pip install --upgrade crawl4ai rich

# Install Playwright browser (Chromium only is enough)
playwright install chromium

# Copy and fill in credentials
cp .env.example .env
# Edit .env — fill in AWS credentials and SerpAPI key
```

#### 5-2. Live URL inference (local)

```bash
conda activate memophish
cd MemoPhishAgent-Reproduction/agent

cp test_urls.txt.example test_urls.txt
# Edit test_urls.txt — add URLs to analyze (one per line)

mkdir -p results
python src/graph.py \
  --agent full_agent \
  --input test_urls.txt \
  --output results/result.json \
  --use-memory false \
  --use-ai-overview false
```

#### 5-3. Offline dataset evaluation (local)

```bash
conda activate memophish
cd MemoPhishAgent-Reproduction/agent

mkdir -p results
python src/test_dataset.py \
  --dataset-root /path/to/datasets/openphish_5000 \
  --output results/openphish_result.json \
  --agent full_agent \
  --use-memory false \
  --use-ai-overview false
```

### 6. Evaluate results

배치 평가 결과 TSV 두 개(피싱/정상)를 받아 분류 지표를 출력합니다.
`folder == "unknown"` 행(에이전트가 자체 크롤한 서브 URL)과 동일 폴더 중복 행은 자동으로 제거합니다.

`--phish-failed` / `--benign-failed`를 추가하면 실패 URL도 집계에 포함됩니다
(피싱 실패 → FN, 정상 실패 → TN).

```bash
# 기본 (TSV만)
python evaluate.py \
  --phish  results/openphish_result.tsv \
  --benign results/tranco_result.tsv

# 실패 URL 포함 (분모를 입력 수에 맞춤)
python evaluate.py \
  --phish  results/openphish_result.tsv \
  --benign results/tranco_result.tsv \
  --phish-failed  results/openphish_result_failed_urls.txt \
  --benign-failed results/tranco_result_failed_urls.txt
```

Sample output:

```text
==================================================
  MemoPhishAgent Evaluation Results
==================================================
  Dataset       phish=10  benign=10  total=20

  [Confusion Matrix]
    TP=8  FP=2
    FN=2  TN=8

  [Classification Metrics]
    Accuracy  : 0.8000  (80.0%)
    Precision : 0.8000
    Recall    : 0.8000
    F1 Score  : 0.8000
==================================================
```

샘플 TSV는 `results/samples/`에 있어 바로 테스트해볼 수 있습니다:

```bash
python evaluate.py \
  --phish  results/samples/sample_openphish.tsv \
  --benign results/samples/sample_tranco.tsv
```

---

## ⚙️ Agent Options

| Option | Default | Description |
| --- | --- | --- |
| `--agent` | — | `determine` / `noimg_agent` / `full_agent` |
| `--use-memory` | `True` | Enable memory-augmented reasoning |
| `--use-ai-overview` | `True` | Pre-filter via Google AI Overview (SerpAPI) |
| `-k` | `5` | Max similar memories to retrieve |
| `--threshold` | `0.60` | Similarity threshold for memory retrieval |
| `--sample` | `None` | Number of URLs to evaluate (takes first N after sorting, omit to use all) |

## 📄 License

See the [original repository](https://github.com/memophishagent/MemoPhishAgent) for license information.

# MemoPhishAgent — Reproduction

> **이 저장소는 아래 논문을 재현하기 위해 원본 [MemoPhishAgent](https://github.com/memophishagent/MemoPhishAgent)를 개인적으로 fork한 것입니다.
> 논문에서 제안하는 시스템을 직접 구현·실행해보며 결과를 검증하는 것이 목적입니다.**

## Paper

**Title:** MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection

**OpenReview:** [openreview.net/forum?id=xKHYnW2tsO](https://openreview.net/forum?id=xKHYnW2tsO)

## Original Repository

[github.com/memophishagent/MemoPhishAgent](https://github.com/memophishagent/MemoPhishAgent)

## Purpose of This Fork

This fork was created to **reproduce the experiments** described in the paper above.
The goal is to verify the reported results and gain a deeper understanding of the system.

Key differences from the original:

- Model backend updated to `us.anthropic.claude-sonnet-4-6` (AWS Bedrock inference profile)
- Uses `ChatBedrockConverse` for inference profile support
- LangChain v1.x import compatibility fixes applied
- S3 private bucket lookups disabled (original authors' private resource)
- `.gitignore` and `.example` config templates added for safe sharing

---

## 🛡️ MemoPhishAgent: Multi-modal Agent-based End‑to‑End Phishing‑URL Detection Platform

MemoPhishAgent is a multi-modal agent-based end‑to‑end framework for collecting and detecting malicious or impersonation URLs at Internet scale.
It leverages memory-augmented reasoning via LangGraph ReAct agents backed by AWS Bedrock.

## 📂 Repository Structure

```text
.
├── agent/                  # Multi‑modal agent detection framework
│   ├── src/                # Agent source code
│   ├── Dockerfile          # Docker image definition
│   ├── requirements.txt    # Python dependencies
│   ├── serpAPI_key.txt.example   # SerpAPI key template
│   └── test_urls.txt.example     # Input URL list template
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites

- Docker (recommended) or Python 3.11+
- AWS credentials with Bedrock access (us-east-1)
- [SerpAPI](https://serpapi.com/) API key

### 2. Setup

```bash
# Clone this repo
git clone https://github.com/ehddy/MemoPhishAgent-Reproduction.git
cd MemoPhishAgent-Reproduction/agent

# Copy and fill in config files
cp serpAPI_key.txt.example serpAPI_key.txt
# Edit serpAPI_key.txt and paste your SerpAPI key

cp test_urls.txt.example test_urls.txt
# Edit test_urls.txt and add URLs to analyze (one per line)

# Configure AWS credentials
aws configure
```

### 3. Run with Docker

```bash
docker build -t memophish-agent .

docker run --rm \
  -v $(pwd)/serpAPI_key.txt:/app/serpAPI_key.txt:ro \
  -v $(pwd)/test_urls.txt:/app/test_urls.txt:ro \
  -v $(pwd)/result.json:/app/result.json \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=us-east-1 \
  memophish-agent \
  python src/graph.py \
    --agent full_agent \
    --input test_urls.txt \
    --output result.json \
    --use-memory False \
    --use-ai-overview False
```

### 4. Run locally

```bash
pip install -r requirements.txt
playwright install chromium

cd src
python graph.py \
  --agent full_agent \
  --input ../test_urls.txt \
  --output ../result.json \
  --use-memory False \
  --use-ai-overview False
```

## ⚙️ Agent Options

| Option | Default | Description |
| --- | --- | --- |
| `--agent` | — | `determine` / `noimg_agent` / `full_agent` |
| `--use-memory` | `True` | Enable memory-augmented reasoning |
| `--use-ai-overview` | `True` | Pre-filter via Google AI Overview (SerpAPI) |
| `-k` | `5` | Max similar memories to retrieve |
| `--threshold` | `0.60` | Similarity threshold for memory retrieval |

## 📄 License

See the [original repository](https://github.com/memophishagent/MemoPhishAgent) for license information.

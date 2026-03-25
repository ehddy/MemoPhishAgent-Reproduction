# MemoPhishAgent — 프로젝트 개요 및 사용 가이드

## 프로젝트 개요

**MemoPhishAgent**는 LLM 에이전트 + 멀티모달 분석을 활용한 피싱 URL 탐지 플랫폼입니다.
텍스트 크롤링, 스크린샷, 이미지 검사, 메모리 시스템을 조합해 URL의 악성 여부를 판정합니다.

- **LLM**: AWS Bedrock (Claude 3 Sonnet, `anthropic.claude-3-sonnet`)
- **임베딩**: AWS Bedrock (`amazon.titan-embed-image-v1`, 1024 dims)
- **에이전트 프레임워크**: LangGraph (ReAct 패턴)
- **웹 크롤러**: crawl4ai (async)
- **벡터 검색**: FAISS

---

## 프로젝트 구조

```
MemoPhishAgent/
├── agent/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── graph.py            # 에이전트 그래프 빌더 + 메인 실행 진입점
│       ├── agent_helpers.py    # ReAct/Deterministic/NoImg 노드 구현
│       ├── tools.py            # 크롤, 스크린샷, 이미지 검사, FAISS 룩업 등 도구
│       ├── memory.py           # 벡터 기반 에이전틱 메모리 시스템
│       ├── prompts.py          # LLM 시스템 프롬프트 모음
│       ├── state.py            # TypedDict 기반 상태 정의
│       ├── utils.py            # AWS 클라이언트, JSON 파싱, Google 검색 유틸
│       ├── callbacks.py        # 토큰/도구 사용량 트래킹 콜백
│       └── baseline_monolithic.py  # 도구 없는 단순 LLM 베이스라인
└── CLAUDE.md
```

---

## 4가지 탐지 모드

| 모드 | 설명 | 속도 | 정확도 |
|------|------|------|--------|
| **monolithic** | 도구 없이 LLM 프롬프트만 사용 | 가장 빠름 | 낮음 |
| **deterministic** | 고정 파이프라인 (크롤→판정→스크린샷→이미지) | 중간 | 중간 |
| **no_img** | ReAct 에이전트, 텍스트만 분석 | 중간 | 중간 |
| **full_agent** | ReAct 에이전트 + 스크린샷 + 이미지 + 메모리 | 느림 | 가장 높음 |

---

## 실행 방법

### 1. Docker 빌드

```bash
cd agent/
docker build -t memophish-agent .
```

### 2. 입력 파일 준비

`urls.txt` 파일에 한 줄에 URL 하나씩 작성:

```
https://example.com
https://suspicious-site.com
```

### 3. 에이전트 실행

```bash
# full_agent (멀티모달 + 메모리)
docker run --rm -it -v "$(pwd)":/work -w /work memophish-agent \
  python src/graph.py \
  --agent full_agent \
  --input urls.txt \
  --output result.json \
  --use-memory True \
  -k 5 \
  --threshold 0.60

# deterministic (고정 파이프라인)
docker run --rm -it -v "$(pwd)":/work memophish-agent \
  python src/graph.py --agent deterministic --input urls.txt --output result.json

# no_img (텍스트 전용 ReAct)
docker run --rm -it -v "$(pwd)":/work memophish-agent \
  python src/graph.py --agent no_img --input urls.txt --output result.json
```

### 4. monolithic 베이스라인

```bash
docker run --rm -it -v "$(pwd)":/work memophish-agent \
  python src/baseline_monolithic.py --input urls.txt --output result.json
```

---

## CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--agent` | 필수 | `full_agent` / `deterministic` / `no_img` |
| `--input` | 필수 | URL 목록 텍스트 파일 경로 |
| `--output` | 필수 | 결과 JSON 파일 경로 |
| `--use-ai-overview` | `True` | Google AI Overview로 빠른 도메인 사전 검사 |
| `--use-memory` | `True` | 메모리 시스템 활성화 (full_agent 전용) |
| `-k` | `5` | 메모리에서 가져올 유사 사례 수 |
| `--threshold` | `0.60` | 메모리 검색 유사도 임계값 |

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `result.json` | 최종 판정 (`{url, malicious, confidence, reason}`) |
| `result_raw.json` | 전체 추론 흔적 (도구 호출 시퀀스 포함) |
| `result_failed_urls.txt` | 처리 실패한 URL 목록 |

---

## 시스템 아키텍처

```
입력 URLs
    ↓
[Google AI Overview 사전 검사] ← --use-ai-overview
    ↓
[Domain FAISS 룩업] ← S3에서 로드
    ↓
[Content FAISS 룩업]
    ↓
[나머지 URL → ReAct 에이전트 or 결정론적 파이프라인]
    ↓
  크롤 → 텍스트 판정 → 스크린샷 검사 → 이미지 검사
    ↓ (최대 3레벨 재귀)
[메모리 검색 & 저장] ← confidence > 4/5 일 때만 저장
    ↓
최종 판정 JSON 출력
```

---

## 메모리 시스템 (`memory.py`)

- **저장**: URL 크롤링 후 키워드 추출 → Bedrock 임베딩 → InMemoryVectorStore 저장
- **검색**: 새 URL 키워드로 k-NN 검색, 임계값 이상이면 과거 사례 재활용
- **다수결 투표**: 유사 사례들의 악성/정상 다수결로 빠른 판정 가능
- **저장 조건**: 신뢰도(confidence) 4/5 초과 시에만 저장 (낮은 확신은 저장하지 않음)

---

## 환경 요구사항

- **AWS 자격증명**: Bedrock 접근용 (`us-east-1` 리전)
- **SerpAPI 키**: `serpAPI_key.txt` 파일에 저장
- **S3 버킷**: 악성 URL 히스토리 데이터 (FAISS 룩업용, 선택사항)
- **Docker**: Playwright가 포함된 Python 이미지 (`mcr.microsoft.com/playwright/python:v1.44.0`)

---

## 핵심 파일 요약

- [src/graph.py](agent/src/graph.py) — 에이전트 그래프 빌드 및 `main()` 실행 진입점
- [src/agent_helpers.py](agent/src/agent_helpers.py) — 각 노드(ReAct 루프, 결정론적 처리) 구현
- [src/tools.py](agent/src/tools.py) — LangChain Tool 객체들 (크롤, 스크린샷, 이미지, FAISS)
- [src/memory.py](agent/src/memory.py) — 벡터 메모리 저장/검색 시스템
- [src/prompts.py](agent/src/prompts.py) — 7개 LLM 시스템 프롬프트
- [src/utils.py](agent/src/utils.py) — AWS 클라이언트, JSON 파서, Google 검색 헬퍼
- [src/state.py](agent/src/state.py) — LangGraph 상태 스키마 정의

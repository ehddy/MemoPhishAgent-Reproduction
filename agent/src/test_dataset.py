"""
TR-OP 데이터셋 배치 테스트 스크립트
이미 수집된 데이터셋(html.txt, shot.png, info.txt)으로 MemoPhishAgent를 평가한다.
실제 URL 접속 없이 로컬 파일만 사용한다.

사용법:
    cd /home/ehddy/agent/MemoPhishAgent/agent/src
    python test_dataset.py \\
        --dataset-root ../../datasets/TR-OP/openphish_5000 \\
        --output results/openphish_result.json \\
        --agent full_agent \\
        --use-memory true \\
        --use-ai-overview false
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from PIL import Image
from pydantic import BaseModel, Field

from agent_helpers import DeterministicNodes, NoImgNodes, ReactNodes
from callbacks import get_default_callbacks, get_token_usage_callbacks
from memory import AgenticMemorySystem, MemoryNodes
from prompts import SYSTEM_SCREEN
from state import ReactURLState, URLState
from tools import AgentTools
from utils import get_bedrock_client, get_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

MAX_CHARS = 5_000
MAX_IMG = 90_000


# ---------------------------------------------------------------------------
# HTML → 마크다운 변환 (crawl4ai의 file:// 프로토콜 사용)
# CrawlContentTool과 완전히 동일한 파싱 파이프라인을 로컬 파일에 적용한다.
# ---------------------------------------------------------------------------

async def html_file_to_markdown(html_path: Path) -> str:
    """
    crawl4ai의 file:// 프로토콜로 로컬 html.txt를 파싱해 마크다운을 반환한다.
    실제 CrawlContentTool이 URL에서 얻는 result.markdown.raw_markdown과 동일한 형식.
    """
    file_url = f"file://{html_path.absolute()}"
    config = CrawlerRunConfig(cache_mode=None, verbose=False)
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(file_url, config=config)
            return result.markdown.raw_markdown
    except Exception as e:
        logging.warning(f"crawl4ai file:// 파싱 실패 ({html_path.name}): {e}")
        # 폴백: 단순 태그 제거
        html_content = html_path.read_text(encoding="utf-8", errors="ignore")
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_content)).strip()


# ---------------------------------------------------------------------------
# shot.png → base64 변환 (256×256 JPEG 압축, CheckScreenshotTool과 동일 로직)
# ---------------------------------------------------------------------------

def encode_shot_png(shot_path: Path) -> str:
    """shot.png를 읽어 256×256 JPEG(quality=60)으로 압축 후 base64 문자열 반환."""
    with open(shot_path, "rb") as f:
        raw = f.read()
    img = Image.open(BytesIO(raw)).convert("RGB").resize((256, 256), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=60, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# URL → 폴더 매핑 구축
# ---------------------------------------------------------------------------

def build_url_folder_map(dataset_root: str) -> Dict[str, Path]:
    """
    지정된 폴더를 스캔해 {url: folder_path} 매핑을 반환한다.
    각 하위 폴더에서 info.txt(URL), html.txt, shot.png를 필요로 한다.
    """
    root = Path(dataset_root)
    url_map: Dict[str, Path] = {}

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if not (folder / "html.txt").exists():
            logging.debug(f"html.txt 없음, 스킵: {folder.name}")
            continue
        if not (folder / "shot.png").exists():
            logging.debug(f"shot.png 없음, 스킵: {folder.name}")
            continue

        info_file = folder / "info.txt"
        if info_file.exists():
            url = info_file.read_text(encoding="utf-8", errors="ignore").strip()
        else:
            # info.txt 없으면 폴더명에서 도메인 추출
            parts = folder.name.split("+")
            domain = parts[1] if len(parts) > 1 else folder.name
            url = f"https://{domain}"

        if url:
            url_map[url] = folder

    logging.info(f"URL 매핑 완료: {root.name} → 총 {len(url_map)}개")
    return url_map


# ---------------------------------------------------------------------------
# 로컬 파일 기반 도구
# ---------------------------------------------------------------------------

class LocalCrawlInput(BaseModel):
    url: str = Field(..., description="The URL of the page to fetch.")
    screenshot: bool = Field(False, description="Whether to include a screenshot.")


class LocalCrawlTool(BaseTool):
    """
    CrawlContentTool 대체: 실제 URL 접속 없이 로컬 html.txt를 읽어 텍스트 반환.
    name을 "crawl_content"로 유지해 LLM이 동일하게 호출할 수 있도록 한다.
    """
    name: str = "crawl_content"
    description: str = (
        "Fetch the page at `url`, return up to the first 5000 chars of its text."
    )
    args_schema: type = LocalCrawlInput
    url_to_folder: Dict[str, Any]

    def __init__(self, url_to_folder: Dict[str, Any]):
        super().__init__(url_to_folder=url_to_folder)

    async def _arun(self, url: str, screenshot: bool = False) -> Dict[str, Any]:
        logging.info(f"📄 [LocalCrawl] Parsing local html.txt via crawl4ai for: {url}")
        folder = self.url_to_folder.get(url)
        if folder is None:
            logging.warning(f"⚠️  URL 매핑 없음: {url}")
            return {"url": url, "text": "No text.", "screenshot": "No image." if screenshot else None}

        html_path = folder / "html.txt"
        # crawl4ai file:// 파싱 (CrawlContentTool과 동일한 마크다운 변환)
        text = (await html_file_to_markdown(html_path))[:MAX_CHARS]
        logging.info(f"✅ [LocalCrawl] {len(text)} chars from {html_path.name}")

        result: Dict[str, Any] = {"url": url, "text": text}
        if screenshot:
            shot_path = folder / "shot.png"
            if shot_path.exists():
                result["screenshot"] = encode_shot_png(shot_path)[:MAX_IMG]
            else:
                result["screenshot"] = "No image."
        return result

    def _run(self, *args, **kwargs):
        raise NotImplementedError("LocalCrawlTool only supports async invocation.")


class LocalScreenshotInput(BaseModel):
    url: str = Field(..., description="URL whose screenshot to analyze.")


class LocalScreenshotTool(BaseTool):
    """
    CheckScreenshotTool 대체: 실제 URL 접속 없이 로컬 shot.png를 읽어 LLM에 전송.
    LLM 호출 로직은 CheckScreenshotTool._arun()과 동일하다.
    """
    name: str = "check_screenshot"
    description: str = (
        "Analyze a base64-encoded screenshot for phishing-site artifacts.  "
        "Returns a JSON dict with keys: "
        "`malicious` (bool), `confidence` (0-5 integer), `notes` (one-sentence summary)."
    )
    args_schema: type = LocalScreenshotInput
    url_to_folder: Dict[str, Any]
    chat: Any

    def __init__(self, url_to_folder: Dict[str, Any], chat: Any):
        super().__init__(url_to_folder=url_to_folder, chat=chat)

    async def _arun(self, url: str) -> Dict[str, Any]:
        logging.info(f"🖼️  [LocalScreenshot] Reading local shot.png for: {url}")
        folder = self.url_to_folder.get(url)
        if folder is None:
            logging.warning(f"⚠️  URL 매핑 없음: {url}")
            return {"malicious": False, "confidence": 0, "notes": "No screenshot available."}

        shot_path = folder / "shot.png"
        if not shot_path.exists():
            return {"malicious": False, "confidence": 0, "notes": "No screenshot available."}

        try:
            compressed_b64 = encode_shot_png(shot_path)
        except Exception as e:
            logging.error(f"❌ [LocalScreenshot] 이미지 처리 실패: {e}")
            return {"malicious": False, "confidence": 0, "notes": "Failed to process screenshot."}

        # LLM 호출 (CheckScreenshotTool._arun() lines 711-745와 동일)
        messages = [
            SystemMessage(content=SYSTEM_SCREEN.content),
            HumanMessage(
                content=[
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": compressed_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze whether this screenshot is a phishing site. "
                            "Respond with ONLY a JSON object, no other text: "
                            '{"malicious": bool, "confidence": int(0-5), "notes": "one sentence"}'
                        ),
                    },
                ]
            ),
        ]

        resp = await self.chat.ainvoke(messages)

        try:
            return json.loads(resp.content)
        except (json.JSONDecodeError, AttributeError):
            pass

        try:
            match = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

        return {"malicious": False, "confidence": 0, "notes": "Failed to parse model response."}

    def _run(self, *args, **kwargs):
        raise NotImplementedError("LocalScreenshotTool only supports async invocation.")


# ---------------------------------------------------------------------------
# 에이전트 빌더 (build_full_agent 기반, 두 도구만 교체)
# ---------------------------------------------------------------------------

def build_dataset_agent(url_to_folder: Dict, args) -> Any:
    """
    graph.py의 build_full_agent()와 동일한 구조지만
    CrawlContentTool → LocalCrawlTool
    CheckScreenshotTool → LocalScreenshotTool 으로 교체한다.
    """
    client = get_bedrock_client()
    callbacks = get_default_callbacks()
    counter = callbacks[1]          # LLM 호출 카운터
    token_callback = get_token_usage_callbacks()
    callbacks.append(token_callback)
    llm = get_llm(client, callbacks)

    # 기존 AgentTools 생성 후 로컬 도구로 교체
    agent_tools = AgentTools(llm)
    agent_tools.crawl = LocalCrawlTool(url_to_folder=url_to_folder)
    agent_tools.check_screenshot = LocalScreenshotTool(url_to_folder=url_to_folder, chat=llm)

    tool_list = [
        agent_tools.crawl,
        # agent_tools.extract_targets,  # 오프라인 평가 시 비활성화: 서브링크 크롤링으로 1 URL → 복수 결과 발생 방지
        agent_tools.check_img,
        agent_tools.check_screenshot,
        agent_tools.serpapi_search,
    ]

    config = RunnableConfig(callbacks=callbacks, recursion_limit=80)

    if args.agent == "determine":
        nodes = DeterministicNodes(agent_tools, token_callback)
        graph = StateGraph(URLState)
        graph.add_node("deterministic_judge", nodes.process)
        graph.add_edge(START, "deterministic_judge")
        graph.add_edge("deterministic_judge", END)
        return graph.compile(), token_callback, counter

    if args.agent == "noimg_agent":
        react_agent_inner = create_react_agent(
            model=llm, tools=[agent_tools.crawl, agent_tools.extract_links]
        )
        nodes = NoImgNodes(react_agent=react_agent_inner, config=config, token_callback=token_callback)
        graph = StateGraph(URLState)
        graph.add_node("judge", nodes.react_judge_node)
        graph.add_edge(START, "judge")
        graph.add_edge("judge", END)
        return graph.compile(), token_callback, counter

    # default: full_agent
    memory_kwargs = {"k": args.k, "threshold": args.threshold}
    agent_memory = AgenticMemorySystem(llm, client, **memory_kwargs)
    memory_nodes = MemoryNodes(agent_tools, agent_memory)
    react_nodes = ReactNodes(
        llm=llm,
        tools=tool_list,
        token_callback=token_callback,
        config=config,
        args=args,
    )

    react_builder = StateGraph(ReactURLState, input=ReactURLState, config_schema=config)
    if args.use_memory:
        react_builder.add_node("prepare_memory", memory_nodes.prepare_memory)
        react_builder.add_node("store_memory", memory_nodes.store_memory)
        react_builder.add_node(react_nodes.call_model)
        react_builder.add_node("tools", ToolNode(tool_list))
        react_builder.add_edge("__start__", "prepare_memory")
        react_builder.add_edge("prepare_memory", "call_model")
        react_builder.add_conditional_edges("call_model", react_nodes.route_model_output)
        react_builder.add_edge("tools", "call_model")
        react_builder.add_edge("store_memory", "__end__")
    else:
        react_builder.add_node("tools", ToolNode(tool_list))
        react_builder.add_node("store_memory", memory_nodes.store_memory)
        react_builder.add_node(react_nodes.call_model)
        react_builder.add_edge("__start__", "call_model")
        react_builder.add_conditional_edges("call_model", react_nodes.route_model_output)
        react_builder.add_edge("tools", "call_model")

    react_agent_compiled = react_builder.compile(name="ReAct Agent")
    react_nodes.react_agent = react_agent_compiled

    graph = StateGraph(URLState)
    graph.add_node("judge", react_nodes.react_judge_node)
    graph.add_edge(START, "judge")
    graph.add_edge("judge", END)
    return graph.compile(), token_callback, counter


# ---------------------------------------------------------------------------
# TSV 결과 저장
# ---------------------------------------------------------------------------

def save_tsv(results: List[Dict], url_to_folder: Dict, tsv_path: str):
    """판별 결과를 TSV 포맷으로 저장한다."""
    os.makedirs(os.path.dirname(tsv_path) or ".", exist_ok=True)
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("folder\turl\tprediction\tconfidence\treason\n")
        for r in results:
            url = r.get("url", "")
            folder = url_to_folder.get(url)
            folder_name = folder.name if folder else "unknown"
            prediction  = "phish" if r.get("malicious") else "benign"
            confidence  = r.get("confidence", "")
            reason      = r.get("reason", "").replace("\n", " ").replace("\t", " ")
            f.write(f"{folder_name}\t{url}\t{prediction}\t{confidence}\t{reason}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="로컬 데이터셋으로 MemoPhishAgent를 배치 평가한다."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="평가할 데이터셋 폴더 (하위에 URL별 폴더가 있어야 함)",
    )
    parser.add_argument(
        "--agent",
        choices=["determine", "noimg_agent", "full_agent"],
        default="full_agent",
    )
    parser.add_argument(
        "--output",
        default="results/result.json",
        help="JSON 결과 저장 경로",
    )
    parser.add_argument(
        "--use-ai-overview",
        default=False,
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        help="Google AI Overview 사전 검사 사용 여부 (오프라인 데이터셋에서는 False 권장)",
    )
    parser.add_argument(
        "--use-memory",
        default=False,
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        help="메모리 시스템 사용 여부",
    )
    parser.add_argument("-k", default=5, type=int, help="메모리 검색 최대 개수")
    parser.add_argument(
        "--threshold", default=0.60, type=float, help="메모리 유사도 임계값"
    )
    parser.add_argument(
        "--sample",
        default=None,
        type=int,
        help="처리할 URL 수 (정렬 후 앞에서 N개, 생략 시 전체 사용)",
    )
    args = parser.parse_args()

    # 출력 파일명에 타임스탬프 삽입 (예: result.json → result_20260522_040648.json)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem, ext = args.output.rsplit(".", 1)
    args.output = f"{stem}_{timestamp}.{ext}"
    output_base = args.output.rsplit(".", 1)[0]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # 1. URL → 폴더 매핑 구축
    url_to_folder = build_url_folder_map(args.dataset_root)
    if not url_to_folder:
        logging.error("처리할 URL이 없습니다. --dataset-root 경로를 확인하세요.")
        raise SystemExit(1)

    urls = list(url_to_folder.keys())

    # 앞에서 N개 자르기 (--sample 지정 시)
    if args.sample and args.sample < len(urls):
        urls = urls[:args.sample]
        logging.info(f"샘플링: 정렬 후 앞 {args.sample}/{len(url_to_folder)}개 선택")

    logging.info(f"총 {len(urls)}개 URL 처리 시작 (agent={args.agent})")
    logging.info(f"결과 저장 경로: {args.output}")

    # 2. 에이전트 빌드
    agent, token_callback, counter = build_dataset_agent(url_to_folder, args)

    # 3. URL 단위 incremental 실행 및 저장
    async def run_incremental():
        json_result: List[Dict] = []
        failed_urls: List[str] = []
        run_start = time.time()

        for i, url in enumerate(urls, 1):
            url_start = time.time()
            result = await agent.ainvoke({"urls": [url]})
            elapsed = round(time.time() - url_start, 2)

            batch = result.get("json_result", [])
            batch_failed = result.get("failed_urls", [])

            for v in batch:
                v["elapsed_sec"] = elapsed
            json_result.extend(batch)
            failed_urls.extend(batch_failed)

            # 루프 중에는 results 배열만 저장 (incremental)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({"results": json_result}, f, indent=2, ensure_ascii=False)
            logging.info(f"[{i}/{len(urls)}] {elapsed}s — 누적 {len(json_result)}건")

        total_elapsed = round(time.time() - run_start, 2)
        return json_result, failed_urls, total_elapsed

    json_result, failed_urls, total_elapsed = asyncio.run(run_incremental())
    avg_elapsed = round(total_elapsed / len(urls), 2) if urls else 0
    logging.info("총 실행 시간: %.2fs  URL당 평균: %.2fs", total_elapsed, avg_elapsed)

    # 완료 후 run_info를 붙여 최종 저장
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": json_result,
                "run_info": {
                    "total_urls": len(urls),
                    "completed": len(json_result),
                    "failed": len(failed_urls),
                    "total_elapsed_sec": total_elapsed,
                    "avg_elapsed_sec": avg_elapsed,
                    "token_usage": token_callback.usage_metadata,
                    "llm_calls": counter.count,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logging.info(f"JSON 결과 저장 (run_info 포함): {args.output}")

    # 4. TSV 저장
    tsv_path = f"{output_base}.tsv"
    save_tsv(json_result, url_to_folder, tsv_path)
    logging.info(f"TSV 결과 저장: {tsv_path}")

    # 5. 실패 URL 저장
    if failed_urls:
        failed_path = f"{output_base}_failed_urls.txt"
        with open(failed_path, "w", encoding="utf-8") as f:
            for u in failed_urls:
                f.write(u + "\n")
        logging.info(f"실패 URL 저장: {failed_path} ({len(failed_urls)}개)")

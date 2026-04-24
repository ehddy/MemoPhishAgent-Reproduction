import base64
import io
import json
import logging
import os
from io import BytesIO
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import boto3
import httpx
import pandas as pd
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from furl import furl
from langchain_core.messages import HumanMessage  # [FIXED] langchain.schema removed in v1.x
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, Tool, tool  # [FIXED] langchain.tools removed in v1.x
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import FAISS
from PIL import Image
from pydantic import BaseModel, Field

from prompts import SYSTEM_EXTRACT, SYSTEM_JUDGE, SYSTEM_JUDGE_IMG, SYSTEM_SCREEN
from state import URLState
from utils import AWS_REGION, find_all_link_urls, find_image_urls, get_bedrock_image_type

import re
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)

MAX_CHARS = 5_000  # keep payload LLM-safe (≈ 1 k-tokens)
MAX_IMG = 90_000
MAX_LINKS = 2
MAX_IMAGES = 2

with open("serpAPI_key.txt", "r") as f:
    api_key = f.read().strip()

serpapi = SerpAPIWrapper(serpapi_api_key=api_key)

def serpapi_search_with_fallback(query: str) -> str:
    """
    Wrapper for SerpAPI that gracefully handles errors when no results are found.
    """
    logging.info(f"🔎 [SerpAPI Tool] Searching: {query}")
    try:
        result = serpapi.run(query)
        logging.info(f"✅ [SerpAPI Tool] Found {len(result)} chars of results")
        return result
    except ValueError as e:
        error_msg = str(e)
        logging.warning(f"⚠️ [SerpAPI Tool] No results found: {error_msg}")
        return f"No search results found for query: {query}. This might indicate an obscure or recently created domain."
    except Exception as e:
        logging.error(f"❌ [SerpAPI Tool] Unexpected error: {e}")
        return f"Search failed with error: {str(e)}"

serpapi_tool = Tool.from_function(
    func=serpapi_search_with_fallback,
    name="serpapi_search",
    description=(
        "Use this to look up facts on the web. "
        "Input: a search query (string). "
        "Output: the combined text of the top search results, or a message if no results found."
    ),
)


class CrawlContentInput(BaseModel):
    """
    Input schema for the `crawl_content` tool.
    """

    url: str = Field(..., description="The URL of the page to fetch.")
    screenshot: bool = Field(
        False, description="Whether to take a screenshot of the page."
    )


class CrawlContentTool(BaseTool):
    name: str = "crawl_content"
    description: str = (
        "Fetch the page at `url`, return up to the first 5000 chars of its text."
    )
    args_schema: type = CrawlContentInput

    async def _arun(self, url: str, screenshot: bool = False) -> Dict[str, Any]:
        logging.info(f"🕷️ [Crawl Tool] Starting crawl for: {url} (screenshot: {screenshot})")

        # 1) Try adding http:// or https:// if missing
        prefixes = ("http://", "https://")
        if any(url.startswith(p) for p in prefixes):
            candidates = [url]
        else:
            candidates = [f"{p}{url}" for p in prefixes]
            logging.info(f"🔗 [Crawl Tool] URL missing protocol, trying: {candidates}")

        # 2) Crawl with JS
        async with AsyncWebCrawler() as crawler:
            js_code = [
                """
                    (() => {
                        const loadMoreButton = Array.from(document.querySelectorAll('button'))
                            .find(button => button.textContent.includes('Load More'));
                        if (loadMoreButton) loadMoreButton.click();
                    })();
                    """
            ]
            config = CrawlerRunConfig(
                cache_mode=None,  # CacheMode.ENABLED
                js_code=js_code,
                verbose=False,
                screenshot=screenshot,
            )

            for candidate in candidates:
                try:
                    logging.info(f"🌐 [Crawl Tool] Attempting to fetch: {candidate}")
                    result = await crawler.arun(candidate, config=config)
                    snippet = result.markdown.raw_markdown[:MAX_CHARS]
                    text_length = len(snippet)
                    logging.info(f"✅ [Crawl Tool] Successfully fetched {text_length} chars from {candidate}")

                    if screenshot:
                        screenshot_truncated = result.screenshot[:MAX_IMG]
                        logging.info(f"📸 [Crawl Tool] Screenshot captured ({len(screenshot_truncated)} bytes)")
                        return {
                            "url": candidate,
                            "text": snippet,
                            "screenshot": screenshot_truncated,
                        }
                    else:
                        return {"url": candidate, "text": snippet}

                except Exception as e:
                    logging.warning(f"❌ [Crawl Tool] Error crawling {candidate}: {e}")
                    continue

        # 3) Fallback if all attempts failed
        logging.error(f"❌ [Crawl Tool] All crawl attempts failed for: {url}")
        if screenshot:
            fallback: Dict[str, Any] = {
                "url": url,
                "text": "No text.",
                "screenshot": "No image.",
            }
        else:
            fallback: Dict[str, Any] = {"url": url, "text": "No text."}
        return fallback

    def _run(self, *args, **kwargs):
        raise NotImplementedError(
            "CrawlContentTool only supports async invocation via `_arun`."
        )


def make_extract_urls_no_images(chat):
    """
    Factory that returns an `extract_urls_no_images` tool bound to `chat`.
    """

    @tool(parse_docstring=True)
    async def extract_urls_no_images(
        url: str,
        text: Optional[str] = "",
    ) -> Dict[str, List[str]]:
        """Given a url and its corresponding content, select which non-image links
        to be crawled next. Image URLs are ignored entirely.

        Args:
            url: The page URL.
            text: The truncated page text.

        Returns:
            A dict with a single key:
            * **to_crawl**: list of URLs (excluding any image URLs) to crawl next.
        """
        max_links = 2
        logging.info(f"extract_urls_no_images received page: {url}")

        if text == "":
            async with AsyncWebCrawler() as crawler:
                # If no text provided, render JS (e.g., “Load More” buttons) and fetch page text
                js_code = [
                    """
                    (() => {
                        const loadMoreButton = Array.from(document.querySelectorAll('button'))
                            .find(button => button.textContent.includes('Load More'));
                        if (loadMoreButton) loadMoreButton.click();
                    })();
                    """
                ]
                config = CrawlerRunConfig(
                    cache_mode=None,
                    js_code=js_code,
                    verbose=False,
                    page_timeout=30000,
                )
                try:
                    result = await crawler.arun(url, config=config)
                    combined = {"markdown": result.markdown.raw_markdown}
                    snippet = combined["markdown"][:MAX_CHARS]
                except Exception as e:
                    logging.info(f"❌ Error crawling {url}: {e}")
                    snippet = "No text"
        else:
            snippet = text

        # Extract all links and identify which ones are image URLs
        img_urls = find_image_urls(snippet)
        all_urls = find_all_link_urls(snippet)

        seen = set()
        non_image_urls = []
        for link in all_urls:
            # Exclude any URL that appears in the image-URL list
            if link not in img_urls and link not in seen:
                seen.add(link)
                non_image_urls.append(link)

        # Build payload with only the non-image links
        payload = {
            "url": url,
            "text": text[:4000],
            "non_image_links": non_image_urls,
        }
        logging.info(f"extract_urls_no_images invoke llm for: {url}")
        # Ask the LLM which of these non-image links to enqueue for crawling
        resp = await chat.ainvoke(
            [
                SYSTEM_EXTRACT,
                HumanMessage(
                    content=(
                        "The page's URL, text snippet, and list of non-image hyperlinks are provided: "
                        f"{payload}"
                    )
                ),
            ]
        )

        # Parse the JSON response and limit to max_links
        try:
            parsed = json.loads(resp.content)
            to_crawl = parsed.get("to_crawl", [])[:max_links]
        except Exception as e:
            logging.info("extract_urls_no_images JSON parse failed: %s", e)
            # Fallback: take the first max_links non-image URLs
            to_crawl = non_image_urls[:max_links]

        return {"to_crawl": to_crawl}

    return extract_urls_no_images


def make_judge_crawled_page(chat):
    """
    Factory that returns a `judge_page` tool bound to the given `chat` LLM.
    """

    @tool
    async def judge_crawled_page(
        url: str,
        text: str,
    ) -> Dict:
        """
        Args:
            url: The page URL.
            text: The truncated page text.

        Returns:
            A dict:
            {
              "url": str,
              "malicious": bool,
              "confidence": int,
              "reason": str
            }
        """
        logging.info("judge_page received URL: %s", url)
        # build the human prompt
        payload = {
            "url": url,
            "text": text[:2000],
        }
        human = HumanMessage(content=json.dumps(payload))
        # call the LLM
        resp = await chat.ainvoke(
            [
                SYSTEM_JUDGE,
                HumanMessage(content=f"Judge if this URL: {url} is malicious or not."),
            ]
        )
        # parse and return
        try:
            return json.loads(resp.content)
        except Exception as e:
            logging.info("judge_page JSON parse failed: %s", e)
            # fallback
            return {
                "url": url,
                "malicious": False,
                "confidence": 0.0,
                "reason": "Failed to parse model response",
            }

    return judge_crawled_page


class ExtractTargetsInput(BaseModel):
    """
    Input schema for the `extract_targets_tool`.
    """

    url: str = Field(..., description="The URL of the page to analyze.")
    text: Optional[str] = Field(
        "",
        description="The page text (truncated) if already fetched; "
        "leave empty to auto-crawl.",
    )


class ExtractTargetsTool(BaseTool):
    name: str = "extract_targets_tool"
    description: str = (
        "Given a URL and its page text, select a small subset of links (`to_crawl`) "
        "and images (`to_check_images`) for deeper inspection."
    )
    args_schema: type = ExtractTargetsInput
    chat: Any

    def __init__(self, chat: Any):
        super().__init__(chat=chat)

    async def _arun(self, url: str, text: Optional[str] = "") -> Dict[str, List[str]]:
        # 1) If no text provided, crawl the page to get it
        snippet = text
        if not snippet:
            async with AsyncWebCrawler() as crawler:
                js_code = [
                    """
                            (() => {
                                const loadMoreButton = Array.from(document.querySelectorAll('button'))
                                    .find(button => button.textContent.includes('Load More'));
                                if (loadMoreButton) loadMoreButton.click();
                            })();
                            """
                ]
                config = CrawlerRunConfig(
                    cache_mode=None,
                    js_code=js_code,
                    verbose=False,
                )
                try:
                    result = await crawler.arun(url, config=config)
                    snippet = result.markdown.raw_markdown[:MAX_CHARS]
                except Exception as e:
                    logging.info(f"❌ Error crawling {url}: {e}")
                    snippet = "No text"

        # 2) Extract raw URLs from the snippet
        img_urls = find_image_urls(snippet)
        all_urls = find_all_link_urls(snippet)

        non_image_urls = []
        seen = set()
        for u in all_urls:
            if u not in img_urls and u not in seen:
                seen.add(u)
                non_image_urls.append(u)

        cleaned_img_urls = []
        for u in img_urls:
            try:
                cleaned_img_urls.append(furl(u).remove(query=True).url)
            except ValueError:
                logging.info(f"Skipping invalid URL: {url} - Error: {e}.")
                continue

        # 3) Ask the LLM which to pick
        payload = {
            "url": url,
            "text": snippet,
            "non image links": non_image_urls,
            "images links": cleaned_img_urls,
        }
        resp = await self.chat.ainvoke(
            [
                SYSTEM_EXTRACT,
                HumanMessage(
                    content=f"The page's URL, text snippet, list of hyperlinks, and list of image URLs are attached here: {payload}"
                ),
            ]
        )

        # 4) Parse or fallback
        try:
            parsed = json.loads(resp.content)
            to_crawl = parsed.get("to_crawl", [])[:MAX_LINKS]
            to_check_images = parsed.get("to_check_images", [])[:MAX_IMAGES]
        except Exception as e:
            logging.info(f"extract_targets JSON parse failed: {e}\n{resp.content}")
            to_crawl = non_image_urls[:MAX_LINKS]
            to_check_images = cleaned_img_urls[:MAX_IMAGES]

        return {"to_crawl": to_crawl, "to_check_images": to_check_images}

    def _run(self, *args, **kwargs):
        raise NotImplementedError(
            "`extract_targets_tool` only supports async invocation."
        )


class CheckImageInput(BaseModel):
    """
    Input schema for the `check_image` tool.
    """

    img_url: str = Field(..., description="The URL of the image to fetch and describe.")


class CheckImageTool(BaseTool):
    name: str = "check_image"
    description: str = (
        "Fetch an image from a URL, send it to the LLM, and return a one-sentence description."
    )
    args_schema: type = CheckImageInput
    chat: Any

    def __init__(self, chat: Any):
        super().__init__(chat=chat)

    # async def _arun(self, img_url: str) -> Dict[str, str]:
    #     image_data = base64.b64encode(httpx.get(img_url).content).decode("utf-8")
    #     response = httpx.get(img_url)
    #     img_type = response.headers["Content-Type"]
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": "Describe this image.",
    #                 },
    #                 {
    #                     "type": "image",
    #                     "source": {
    #                         "type": "base64",
    #                         "media_type": img_type,
    #                         "data": image_data,
    #                     },
    #                 },
    #             ],
    #         }
    #     ]
    #     resp = await self.chat.ainvoke(messages)

    #     return {"image_url": img_url, "description": resp.content}

    async def _arun(self, img_url: str) -> Dict[str, str]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(img_url, follow_redirects=True)
            resp.raise_for_status()
            img_bytes = resp.content

        # 실제 바이너리 데이터를 분석하여 타입 결정
        img_type = get_bedrock_image_type(img_bytes)
        image_b64 = base64.b64encode(img_bytes).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img_type, # 동적으로 감지된 타입 사용
                            "data": image_b64,
                        },
                    },
                ],
            }
        ]
        resp_llm = await self.chat.ainvoke(messages)
        return {"image_url": img_url, "description": resp_llm.content}
    

    def _run(self, *args, **kwargs):
        raise NotImplementedError(
            "`check_image` only supports async invocation via `_arun`."
        )


def make_judge_image(chat):
    """
    Factory that returns a `judge_image` tool bound to the given `chat` LLM.
    This tool takes a purely textual image description (returned by `check_image`)
    and decides if that image indicates phishing content.
    """

    @tool(parse_docstring=True)
    async def judge_image(image_url: str, description: str) -> Dict:
        """
        Given an image URL and its textual description, decide if it is
        part of a phishing site.

        Args:
            image_url: The source URL of the image.
            description: A natural-language description of what appears in the image.

        Returns:
            A dict with:
            * **url** (str): the same `image_url`
            * **malicious** (bool): True if the image likely indicates phishing
            * **confidence** (integer): 0-5 how sure the model is
            * **reason** (str): one-sentence rationale for the decision
        """
        # System prompt: frame the task

        # Human message: supply the URL and its description
        human_msg = HumanMessage(
            content=json.dumps({"image_url": image_url, "description": description})
        )

        # Invoke the LLM
        resp = await chat.ainvoke([SYSTEM_JUDGE_IMG, human_msg])

        # Parse JSON response (fallback to safe default)
        try:
            parsed = json.loads(resp.content)
            return parsed
        except Exception:
            return {
                "url": image_url,
                "malicious": False,
                "confidence": 0.0,
                "reason": "Failed to parse model response",
            }

    return judge_image


def build_lookbook_retriever_from_s3(
    bucket: str,
    k: int = 3,
    score_threshold: float = 0.8,
    faiss_path: str = "faiss_url_db/",
    prefix: str = "Proactive_discovery/",
    key: str = "takedown.csv",
):
    """Load CSV from S3, filter, and build a FAISS retriever."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    boto3_bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    if "content" in faiss_path:
        bedrock_embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0", client=boto3_bedrock
        )
    else:
        bedrock_embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1", client=boto3_bedrock
        )
    if os.path.exists(faiss_path):
        logging.info(f"Loading FAISS index from {faiss_path}.")
        vectordb = FAISS.load_local(
            faiss_path, bedrock_embeddings, allow_dangerous_deserialization=True
        )
    else:
        logging.info("FAISS index not found. Building from scratch...")
        # 1. fetch CSV
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = response.get("Contents", [])
        csv_keys = [obj["Key"] for obj in contents if obj["Key"].endswith(".csv")]

        obj = s3.get_object(
            Bucket=bucket,
            Key=key,
        )
        df_takedown = pd.read_csv(io.BytesIO(obj["Body"].read()))["Attack URL"]

        docs = []
        dfs = []

        for key in csv_keys:
            if "urls_w_label" not in key:
                continue
            obj = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            # 2. filter rows where takedown_url_id is not null
            df = df[df["takedown_url_id"].notna()]
            if not df.empty:
                dfs.append(df["url"])

        df_combined = pd.concat(dfs, ignore_index=True)
        final_df = pd.concat([df_combined, df_takedown], ignore_index=True)
        domains = []
        for url in final_df:
            domain = urlparse(url).netloc
            domains.append(domain)
        domains = list(set(domains))
        # 3. turn into LangChain Documents
        for d in domains:
            docs.append(
                Document(
                    page_content=d,
                    metadata={
                        "content": "",  # stash for later
                    },
                )
            )
        # 3a. Save locally
        # final_df.to_csv('take_down_urls.csv', index=False)
        logging.info(f"📁 Scanned {len(csv_keys)} CSV files from S3.")
        logging.info(f"✅ Added {len(domains)} URLs to the vector store.")

        # 4. embed & index
        vectordb = FAISS.from_documents(docs, bedrock_embeddings)
        os.makedirs(faiss_path, exist_ok=True)
        vectordb.save_local(faiss_path)
        logging.info(f"FAISS index saved to {faiss_path}")

    # after loading
    logging.info("FAISS index dimension: %s", str(vectordb.index.d))

    # after embedding
    vec = bedrock_embeddings.embed_query("test query")
    logging.info("Embedding output dimension: %s", str(len(vec)))

    return vectordb.as_retriever(
        search_kwargs={"k": k, "score_threshold": score_threshold}
    )


class CheckScreenshotInput(BaseModel):
    """
    Input schema for the `check_screenshot` tool.
    """

    url: str = Field(
        ...,
        description="URL(string) that you want to extract screenshot and analyze for phishing artifacts.",
    )


class CheckScreenshotTool(BaseTool):
    name: str = "check_screenshot"
    description: str = (
        "Analyze a base64-encoded screenshot for phishing-site artifacts.  "
        "Returns a JSON dict with keys: "
        "`malicious` (bool), `confidence` (0-5 integer), `notes` (one-sentence summary)."
    )
    args_schema: type = CheckScreenshotInput
    chat: Any

    def __init__(self, chat: Any):
        """
        `chat` must be an LLM with `.ainvoke(messages)` that
        returns a text response containing valid JSON.
        """
        super().__init__(chat=chat)

    async def _arun(self, url: str) -> Dict[str, Any]:
        logging.info(f"📸 [Screenshot Tool] Capturing screenshot for: {url}")

        prefixes = ["http://", "https://"]
        if any(url.startswith(p) for p in prefixes):
            candidates = [url]
        else:
            candidates = [f"{p}{url}" for p in prefixes]

        async with AsyncWebCrawler() as crawler:

            js_code = [
                """
                        (() => {
                            const loadMoreButton = Array.from(document.querySelectorAll('button'))
                                .find(button => button.textContent.includes('Load More'));
                            if (loadMoreButton) loadMoreButton.click();
                        })();
                        """
            ]
            config = CrawlerRunConfig(
                cache_mode=None,
                js_code=js_code,
                verbose=False,
                screenshot=True,
            )
            for candidate_url in candidates:
                try:
                    logging.info(f"📷 [Screenshot Tool] Attempting screenshot of: {candidate_url}")
                    result = await crawler.arun(candidate_url, config=config)
                    screenshot = result.screenshot
                    logging.info(f"✅ [Screenshot Tool] Screenshot captured successfully")
                    break

                except Exception as e:
                    logging.warning(f"❌ [Screenshot Tool] Error capturing {candidate_url}: {e}")
                    continue
        try:
            logging.info(f"🖼️ [Screenshot Tool] Processing and compressing screenshot...")
            raw = base64.b64decode(screenshot)
            img = Image.open(BytesIO(raw)).resize((256, 256), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=60, optimize=True)
            compressed_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Step 2: Build messages with proper LangChain message types
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
        except Exception as e:
            logging.error(f"❌ [Screenshot Tool] Image processing failed: {e}")
            return {
                "url": url,
                "malicious": False,
                "confidence": 0.0,
                "reason": "Failed to parse screenshot",
            }

        logging.info(f"🤖 [Screenshot Tool] Sending screenshot to LLM for analysis...")
        resp = await self.chat.ainvoke(messages)

        # Step 3: parse JSON or return safe fallback
        try:
            result = json.loads(resp.content)
            logging.info(f"✅ [Screenshot Tool] Analysis complete - Malicious: {result.get('malicious')}, Confidence: {result.get('confidence')}/5")
            return result
        except (json.JSONDecodeError, AttributeError):
            pass

        # Step 3-1: Extract JSON block when mixed with surrounding text
        try:
            match = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if match:
                result = json.loads(match.group())
                logging.info(f"✅ [Screenshot Tool] Analysis complete - Malicious: {result.get('malicious')}, Confidence: {result.get('confidence')}/5")
                return result
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

        logging.warning(f"⚠️ [Screenshot Tool] Failed to parse LLM response: {resp.content[:300]}")
        return {
            "malicious": False,
            "confidence": 0,
            "notes": "Failed to parse model response.",
        }

    def _run(self, *args, **kwargs):
        # if someone calls sync, you can either:
        # • raise NotImplementedError
        # • or run the async code in an event loop
        raise NotImplementedError("Please use async invocation (_arun).")


class AgentTools:
    def __init__(self, llm: Any):
        # init tools that need llm
        self.crawl = CrawlContentTool()
        self.extract_links = make_extract_urls_no_images(llm)
        self.judge_crawled_page = make_judge_crawled_page(llm)
        self.check_screenshot = CheckScreenshotTool(llm)
        self.extract_targets = ExtractTargetsTool(llm)
        self.check_img = CheckImageTool(llm)
        self.judge_img = make_judge_image(llm)
        self.serpapi_search = serpapi_tool
        # historicals - [DISABLED] requires private S3 bucket "guardians-cipher" (authors' internal bucket)
        # self.domain_lookup = build_lookbook_retriever_from_s3(
        #     "guardians-cipher",
        #     k=3,
        #     score_threshold=0.2,
        #     faiss_path="faiss_url_domain_db",
        # )
        # self.content_lookup = build_lookbook_retriever_from_s3(
        #     "guardians-cipher",
        #     k=3,
        #     score_threshold=0.5,
        #     faiss_path="faiss_url_content_db",
        # )
        self.domain_lookup = None
        self.content_lookup = None

    async def lookup_malicious_url_tool(self, url: str) -> str:
        """
        Returns up to 3 historically-flagged malicious URLs similar to the given URL's domain name.
        Call sparingly when you need extra evidence. Returns 'NO_MATCH' if none.
        """
        docs = self.domain_lookup.get_relevant_documents(url)
        return "NO_MATCH" if not docs else "\n".join(d.page_content for d in docs)

    async def lookup_malicious_url_content_tool(self, text: str) -> str:
        """
        Returns up to 3 historically-flagged malicious URLs similar to the given URL's content.
        Returns 'NO_MATCH' if none.
        """
        docs = self.content_lookup.get_relevant_documents(text)
        return "NO_MATCH" if not docs else "\n".join(d.page_content for d in docs)

    async def lookup_domain_node(
        self, state: URLState
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Splits URLs into:
        • domain_matched: those with a historical match on network location part (also known as the domain name or authority) of a URL.
        • remaining_urls: those without match.
        """
        logging.info("Initial URL domain name match.")
        initial, remaining = [], []
        for url in state["urls"]:
            # Add url head if needed
            if not url.startswith("https://"):
                full_url = f"https://{url}"
            else:
                full_url = url
            url_domain = urlparse(full_url).netloc
            if not url_domain:
                logging.info("Empty domain.")
                remaining.append(url)
                continue
            match = await self.lookup_malicious_url_tool(url)
            if match != "NO_MATCH":
                initial.append(url)
            else:
                remaining.append(url)
        logging.info(
            f"Number of domain-matched URL: {len(initial)}, remaining URL: {len(remaining)}"
        )
        return {"domain_matched": initial, "remaining_urls": remaining}

    async def lookup_content_node(
        self, state: URLState
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Splits URLs into:
        • content_matched: those with a historical match regarding the page content.
        • remaining_urls: those without
        """
        logging.info("URL content pattern match.")
        logging.info(f"URLs after domain match: {len(state['remaining_urls'])}")
        middle, remaining = [], []
        for url in state["remaining_urls"]:
            # call the async retriever
            url_content = await self.crawl.arun(url)
            match = await self.lookup_malicious_url_content_tool(url_content["text"])
            if match != "NO_MATCH":
                middle.append(url)
                logging.info("Match content found.")
            else:
                logging.info("No match content found.")
                remaining.append(url)
        logging.info(
            f"URLs domain matched: {len(state['domain_matched'])}, URLs content matched: {len(middle)}, remaining URL: {len(remaining)}."
        )
        logging.info(f"Content matched URLs: {middle}")
        return {"content_matched": middle, "final_remaining_urls": remaining}

import json
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import boto3
import tldextract
from botocore.config import Config
from langchain_aws.chat_models import ChatBedrock
from serpapi import GoogleSearch

# AWS & Model configuration
AWS_REGION = "us-east-1"
# MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"  # [LEGACY] blocked for new users
MODEL_ID = "global.anthropic.claude-sonnet-4-6"
with open("serpAPI_key.txt", "r") as f:
    API_KEY = f.read().strip()


def get_bedrock_client() -> Any:
    """Instantiate the AWS Bedrock client."""
    config = Config(
        read_timeout=2000,
        retries={
            "max_attempts": 5,  # total tries
            "mode": "adaptive",  # AWS will back off *for you* based on load
        },
    )
    return boto3.client("bedrock-runtime", region_name=AWS_REGION, config=config)


def get_llm(client: Any, callbacks: Optional[List[Any]] = None) -> ChatBedrock:
    """Instantiate the Bedrock-backed LLM."""
    return ChatBedrock(client=client, model_id=MODEL_ID, callbacks=callbacks)


def extract_json_from_llm_output(output: str) -> Dict:
    """
    Extracts and parses a JSON object from the output of an LLM.

    This function handles both of the following cases:
    1. JSON wrapped inside a markdown-style code block (e.g., ```json ... ```).
    2. Raw JSON objects directly in the text (e.g., {...}).

    Args:
        output (str): The full string output from an LLM.

    Returns:
        dict or None: The parsed JSON object as a Python dictionary if found and valid;
        None if no JSON is found or if parsing fails.
    """
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL)
    parsed = {
        "url": "None",
        "malicious": False,
        "confidence": 0.0,
        "reason": "Could not parse model response",
    }
    if not match:
        # Fallback: try to find raw JSON starting with `{` and ending with `}`
        match = re.search(r"(\{.*\})", output, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return parsed
    else:
        print("No JSON found in the output.")
        return parsed


def extract_and_fix(text: str) -> Any:
    """
    Extract all JSON snippets from a text, attempt to parse each.
    """
    results = []
    i = 0
    L = len(text)
    while True:
        start = text.find("{", i)
        if start < 0:
            break
        depth = 0
        for j in range(start, L):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : j + 1]
                    # collapse ANY newline inside the JSON snippet
                    snippet = snippet.replace("\n", " ")
                    try:
                        results.append(json.loads(snippet))
                    except json.JSONDecodeError:
                        pass
                    i = j + 1
                    break
        else:
            break
    return results


def find_image_urls(markdown_text):
    """
    Extract markdown image URLs ending in .jpg/.png so we can exclude them.
    """
    img_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
    urls = img_pattern.findall(markdown_text)
    return {u for u in urls if re.search(r"\.(jpg|png)(?:\?|$)", u, re.IGNORECASE)}


def find_all_link_urls(markdown_text):
    link_pattern = re.compile(r"\[.*?\]\((.*?)\)")
    return link_pattern.findall(markdown_text)


# Google ai overview related
# Having those domains does not gurantee this website is safe, they need further investigation.
SKIP_DOMAINS = {
    "sites.google.com",
    "github.io",
    "gitlab.io",
    "netlify.app",
}


def should_skip(url):
    hostname = urlparse(url).netloc.lower()
    return any(hostname.endswith(d) for d in SKIP_DOMAINS)


def extract_domain_and_brand(url):
    parsed = urlparse(url)
    hostname = parsed.netloc or parsed.path  # support no-scheme lines
    ext = tldextract.extract(hostname)
    brand = ext.domain.replace("-", " ").replace("_", " ").title()
    return hostname, brand


def make_queries(domain, brand):
    return [
        f"{domain} overview",
        f"{brand} site:{domain}",
        f"site:{domain} overview",
        f'"{domain} phishing"',
        f"info:{domain}",
        f"related:{domain}",
        f"info:{domain} scam",
        f"related:{domain} scam",
        f"link:{domain} scam",
        f"link:{domain} phishing",
    ]


def google_search_with_retry(q, api_key, max_retries=5, delay_seconds=3):
    """Perform a Google Search, retrying up to max_retries on any exception.
    """
    params = {
        "engine": "google",
        "q": q,
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "api_key": api_key,
    }

    for attempt in range(1, max_retries + 1):
        try:
            return GoogleSearch(params).get_dict()
        except Exception as e:
            print(f"Attempt {attempt} failed: {e!r}")
            time.sleep(delay_seconds)

    print("All retries exhausted.")
    return {"ai_overview": None}


def fetch_ai_overview_for_query(q):
    # initial search to get a page_token
    resp = google_search_with_retry(q, API_KEY)
    ai_block = resp.get("ai_overview", {})
    token = ai_block.get("page_token")
    if not token:
        return None

    # fetch the detailed overview
    resp2 = GoogleSearch(
        {"engine": "google_ai_overview", "page_token": token, "api_key": API_KEY}
    ).get_dict()

    try:
        return resp2.get("ai_overview", None)
    except:
        print(resp2)
        print("No ai overview available.")
        return None


def extract_text(ai_overview):
    snippets = []
    for block in ai_overview.get("text_blocks", []):
        t = block.get("type")
        if t == "paragraph":
            snippets.append(block.get("snippet", ""))
        elif t == "list":
            for item in block.get("list", []):
                if item.get("snippet"):
                    snippets.append(item["snippet"])
    return "\n".join(snippets)


def analyze_with_llm(llm, text, url):
    system = {
        "role": "system",
        "content": (
            """
           You are an AI security analyst. The following text snippets were obtained by performing Google searches on the domain of the target URL. 
           Using only this information, determine whether the URL is benign or malicious. 
           Output only the raw JSON object, with no markdown, no code fences.
           Respond in JSON format:
                - \"url\": the original url (provided by the user)
                - \"malicious\": true or false  
                - \"confidence\": integer from 0 (lowest) to 5 (highest)  
                - \"reason\": one concise sentence explaining your judgment
        """
        ),
    }
    user = {
        "role": "user",
        "content": (f"URL is: {url}. \n Google serach text snippet:\n{text}"),
    }
    try:
        resp = llm.invoke([system, user])
        return json.loads(resp.content)
    except Exception as e:
        print(f"Error {e}, return None")
        return None


def ai_overview_preprocess(url, llm):
    if should_skip(url):
        # leave those "do-it-yourself" hosting platforms and URL-shortening services for agent judgment.
        return None
    # try each query until we get an ai_overview
    domain, brand = extract_domain_and_brand(url)
    for q in make_queries(domain, brand):
        overview = fetch_ai_overview_for_query(q)
        if overview:
            # only now do we extract text and call the LLM
            text = extract_text(overview)
            judgment = analyze_with_llm(llm, text, url)
            # include URL and query for traceability
            return judgment
    # no overview → skip
    return None


# 새로 추가
def get_bedrock_image_type(image_bytes: bytes) -> str:
    """이미지 바이너리를 체크하여 Bedrock이 지원하는 MIME 타입을 반환합니다."""
    if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        return "image/png"
    elif image_bytes.startswith(b'\xff\xd8'):
        return "image/jpeg"
    elif image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
        return "image/gif"
    elif image_bytes.startswith(b'RIFF') and image_bytes[8:12] == b'WEBP':
        return "image/webp"
    return "image/jpeg" # 기본값
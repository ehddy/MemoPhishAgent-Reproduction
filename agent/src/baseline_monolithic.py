import argparse
import asyncio
import json
import os
import time

import boto3
import botocore.exceptions
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from langchain_aws.chat_models import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from callbacks import get_token_usage_callbacks
from utils import MODEL_ID


async def crawl_and_judge(urls, llm, max_text_chars=2000, max_screenshot_chars=100000):
    """Crawls a list of URLs and uses an LLM to determine if each URL is malicious.

    Each URL is processed by crawling its page, extracting raw markdown and a screenshot,
    and then passing these to the LLM to determine if the site is phishing.

    Args:
        urls (List[str]): List of URLs to be judged.
        llm (BaseLanguageModel): The LLM instance (e.g., ChatBedrock) used to evaluate each site.
        max_text_chars (int, optional): Maximum number of characters from page text to include. Defaults to 2000.
        max_screenshot_chars (int, optional): Maximum number of characters from base64 screenshot. Defaults to 100000.

    Returns:
        List[str]: A list of URLs flagged as malicious.
    """
    malicious_urls = []

    async with AsyncWebCrawler() as crawler:
        for idx, url in enumerate(urls):
            if not url:
                print(f"❌ Skipping record {idx+1}: Empty URL")
                continue

            print(f"[{idx+1}/{len(urls)}] Processing: {url}")
            try:
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
                    cache_mode=CacheMode.ENABLED,
                    js_code=js_code,
                    verbose=False,
                    screenshot=True,
                )
                result = await crawler.arun(url=url, config=config)

                try:
                    raw_markdown = result.markdown.raw_markdown or ""
                except AttributeError:
                    raw_markdown = ""

                screenshot_b64 = result.screenshot or ""

                # Truncate content to fit LLM context
                text_snippet = raw_markdown[:max_text_chars]
                screenshot_snippet = screenshot_b64[:max_screenshot_chars]

                # Build prompt messages using SystemMessage and HumanMessage
                messages = [
                    SystemMessage(
                        content="""
                                You are a cybersecurity analyst. You will receive for each URL:
                                1. The URL string
                                2. A snippet of the page text (up to 2000 chars)
                                3. A base64 screenshot of the page

                                Based on all of this, decide if the site is an phishing attack.
                                Return JSON exactly:

                                {
                                "url": "<the URL>",
                                "malicious": <true|false>,
                                "confidence": <0/1/2/3/4/5>,
                                "reason": "<one-sentence rationale>"
                                }
                            """
                    ),
                    HumanMessage(
                        content=f"""
                                    Judge this URL with the following crawled data:
                                    URL: {url}
                                    Page text snippet:
                                        {text_snippet}
                                    Screenshot (base64, truncated):
                                        {screenshot_snippet}
                                """
                    ),
                ]

                ai_msg = llm.invoke(messages)
                print(ai_msg.content)

                # Parse the LLM response
                try:
                    verdict = json.loads(ai_msg.content)
                    if verdict.get("malicious"):
                        malicious_urls.append(url)
                except json.JSONDecodeError:
                    print(f"❌ Failed to parse LLM response for {url}")

            except botocore.exceptions.ClientError as e:
                error_code = e.response["Error"]["Code"]
                error_message = e.response["Error"]["Message"]

                if (
                    error_code == "ThrottlingException"
                    and "Too many tokens" in error_message
                ):
                    print("Throttling detected: sleeping 3 seconds...")
                    await asyncio.sleep(3)
                else:
                    raise  # re-raise if it's a different ClientError

    return malicious_urls


async def main_async(input_path, output_path):
    """Main async entry point: loads URLs, performs judgment, and writes results.

    This function reads input URLs from a file, initializes the Bedrock client and LLM,
    calls `crawl_and_judge` to identify malicious URLs, and saves them to an output file.

    Args:
        input_path (str): Path to the input file containing one URL per line.
        output_path (str): Path to save the list of malicious URLs.
    """
    # Read URLs
    with open(input_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    # Setup Bedrock client
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )
    token_callback = get_token_usage_callbacks()
    llm = ChatBedrock(
        client=bedrock_runtime,
        model_id=MODEL_ID,
        beta_use_converse_api=False,
        callbacks=[token_callback],
    )

    # Crawl and judge
    start_all = time.perf_counter()
    malicious = await crawl_and_judge(urls, llm)
    total_time = time.perf_counter() - start_all
    avg_time = total_time / len(urls) if urls else 0

    # Save malicious URLs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        for line in malicious:
            file.write(line + "\n")

    # Print summary
    print(
        f"Processed {len(urls)} URLs. Found {len(malicious)} malicious. Avg judge time: {avg_time:.2f}s URL"
    )
    print(f"Token usage: {token_callback.usage_metadata}")


def main():
    parser = argparse.ArgumentParser(
        description="Monolithic baseline for phishing detection"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file with URLs, one per line",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to write JSON of malicious URLs"
    )
    args = parser.parse_args()

    asyncio.run(main_async(args.input, args.output))


if __name__ == "__main__":
    main()

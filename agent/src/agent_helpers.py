import asyncio
import json
import logging
import time
from typing import Any, Literal, cast

import botocore
import botocore.exceptions
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from langchain_aws.chat_models import ChatBedrock
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langchain_core.callbacks import UsageMetadataCallbackHandler

from prompts import SYSTEM_NO_IMG, SYSTEM_REACT, SYSTEM_REACT_MEM
from state import ReactURLState, URLState
from tools import AgentTools
from utils import ai_overview_preprocess, extract_and_fix

logger = logging.getLogger(__name__)


class ReactNodes:
    def __init__(
        self,
        llm: ChatBedrock,
        tools: list[BaseTool],
        token_callback: Any,
        config: RunnableConfig,
        args: Any,
    ):
        self.llm = llm
        self.my_tools = tools
        self.token_callback = token_callback
        self.config = config
        self.args = args
        self.SYSTEM_REACT = SYSTEM_REACT
        self.SYSTEM_REACT_MEM = SYSTEM_REACT_MEM
        # placeholder for the compiled agent
        self.react_agent = None

    async def call_model(self, state: ReactURLState) -> dict[str, list[AIMessage]]:
        model = self.llm.bind_tools(self.my_tools)
        system_message = self.SYSTEM_REACT
        logger.info(f"Find related memory: {state.memory_snippet}")

        # memory‐based override
        if state.memory_majority:
            logging.info(f"Majority of the memories are malicious")
            verdict = {
                "url": state.url,
                "malicious": True,
                "confidence": 5,
                "reason": "Reused majority-vote from past similar URLs (>50% malicious).",
            }
            answer = AIMessage(
                content=json.dumps({"verdicts": [verdict]}),
            )
            return {"messages": [answer]}

        if state.memory_snippet:
            system_message = (
                f"{state.memory_snippet}\n"
                "First leverage those past-case summaries ('memory'), only invoke tools if you need more evidence.\n"
                + self.SYSTEM_REACT_MEM
            )

        # build prompt
        if not state.messages:
            human = {
                "role": "user",
                "content": f"Judge if this URL {state.url} is malicious or phishing site.",
            }
            prompt = [{"role": "system", "content": system_message}, human]
        else:
            prompt = [{"role": "system", "content": system_message}, *state.messages]

        response = cast(
            AIMessage,
            await model.ainvoke(prompt),
        )
        # disable screenshot for clarity
        if response.tool_calls and response.tool_calls[0]["name"] == "crawl_content":
            response.tool_calls[0]["args"]["screenshot"] = False

        # out of steps
        if state.is_last_step and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, I could not find an answer to your question in the specified number of steps.",
                    )
                ]
            }
        return {"messages": [response]}

    def route_model_output(
        self, state: ReactURLState
    ) -> Literal["store_memory", "tools", "__end__"]:
        last_msg = state.messages[-1]
        if not isinstance(last_msg, AIMessage):
            raise ValueError(f"Expected AIMessage, got {type(last_msg).__name__}")

        # still wants a tool?
        if last_msg.tool_calls:
            return "tools"

        # final answer — decide where to land
        return "store_memory" if state.use_memory else "__end__"

    async def react_judge_node(self, state: URLState) -> dict[str, Any]:
        if self.react_agent is None:
            raise RuntimeError("react_agent not set on ReactNodes")

        verdicts, jsons, failed_urls = [], [], []
        for i, url in enumerate(state["urls"]):
            if self.args.use_ai_overview:
                ai_overview_res = ai_overview_preprocess(url, self.llm)
                if ai_overview_res and ai_overview_res["malicious"]:
                    ai_overview_res["memory_case"] = "google_ai_overview"
                    jsons.append(ai_overview_res)
                    continue

            agent_input = {
                "messages": [
                    HumanMessage(
                        content=f"Judge if this URL: {url} is malicious or a phishing website."
                    )
                ],
                "url": url,
                "use_memory": self.args.use_memory,
                "tool_sequence": [],
                "keywords": [],
            }
            tool_sequence = []
            try:
                # Run the async ReAct agent
                async for step in self.react_agent.astream(
                    agent_input, config=self.config, stream_mode="values"
                ):
                    last_msg = step["messages"][-1]
                    step["messages"][-1].pretty_print()

                    # record any tool_calls on this LLM message
                    if getattr(last_msg, "tool_calls", None):
                        for call in last_msg.tool_calls:
                            step["tool_sequence"].append(call["name"])
                            tool_sequence.append(call["name"])

                # pick up memory_case
                final_memory_case = step.get("memory_case", " ")
                final_msg = last_msg.content
                logging.info("===" * 50)
                verdicts.append({"url": url, "reason": final_msg})
                final_json = extract_and_fix(final_msg)
                try:
                    for v in final_json[0]["verdicts"]:
                        v["memory_case"] = final_memory_case
                        jsons.append(v)
                except Exception as e:
                    logging.info(f"Error {e}")
                    failed_urls.append(url)
                    continue

                # every 20 URLs, flush out the file
                if (i + 1) % 20 == 0:
                    with open(self.args.output, "w") as f:
                        json.dump(jsons, f, indent=2)

            except botocore.exceptions.ClientError as e:
                error_code = e.response["Error"]["Code"]
                error_message = e.response["Error"]["Message"]
                failed_urls.append(url)
                with open(
                    f"{self.args.output.split('.')[0]}_failed_urls.txt", "w"
                ) as file:
                    for line in failed_urls:
                        file.write(line + "\n")

                if (
                    error_code == "ThrottlingException"
                    and "Too many tokens" in error_message
                ):
                    logging.warning("Throttling detected: sleeping 60 seconds...")
                    await asyncio.sleep(60)
                else:
                    logging.warning(f"❌ Error when judging {url}: {e}, continue")

        logging.info(f"Token usage: {self.token_callback.usage_metadata}")
        return {"result": verdicts, "json_result": jsons, "failed_urls": failed_urls}


class DeterministicNodes:
    """A deterministic pipeline to classify URLs as malicious or not using a sequence of tools.

    This class uses a provided `AgentTools` instance to crawl pages, judge content,
    check screenshots, and inspect embedded images and linked pages. It collects
    any URLs deemed malicious through various stages and returns final verdict.

    Args:
        tools (AgentTools): Set of async tools for crawling, judging, and extracting.
        token_callback (UsageMetadataCallbackHandler): Callback for logging or tracking token usage.
    """

    def __init__(self, tools: AgentTools, token_callback: UsageMetadataCallbackHandler):
        self.tools = tools
        self.token_callback = token_callback

    async def process(self, state: URLState) -> dict[str, list[str]]:
        """Process the URLs and classify them as malicious or not."""
        page_malicious = []
        screenshot_malicious = []
        final_malicious = []
        failed_urls = []
        for url in state["urls"]:
            try:
                page = await self.tools.crawl.arun({"url": url, "screenshot": True})
                content_judge = await self.tools.judge_crawled_page.arun(
                    {"url": url, "text": page["text"]}
                )
                if content_judge["malicious"]:
                    page_malicious.append(url)
                    logging.info("Page content is malicious.")
                    continue

                ss_judge = await self.tools.check_screenshot.arun(url)
                if ss_judge["malicious"]:
                    screenshot_malicious.append(url)
                    logging.info("Page screenshot is malicious.")
                    continue

                # look one step furter
                targets = await self.tools.extract_targets.arun(
                    {"url": url, "text": page["text"]}
                )
                for img_url in targets["to_check_images"]:
                    try:
                        img_desp = await self.tools.check_img.arun(img_url)
                        img_judge = await self.tools.judge_img.arun(
                            {
                                "image_url": img_desp["image_url"],
                                "description": img_desp["description"],
                            }
                        )
                        if img_judge["malicious"]:
                            screenshot_malicious.append(url)
                            logging.info("Inside image is malicious.")
                            continue
                    except Exception as e:
                        logging.info(f"Error when call image tools: {e}, continue")
                        continue

                for inside_url in targets["to_crawl"]:
                    page = await self.tools.crawl.arun(inside_url)
                    content_judge = await self.tools.judge_crawled_page.arun(
                        {"url": inside_url, "text": page["text"]}
                    )
                    if content_judge["malicious"]:
                        page_malicious.append(url)
                        logging.info("Inside URL is malicious.")
                        continue

            except botocore.exceptions.ClientError as e:
                error_code = e.response["Error"]["Code"]
                error_message = e.response["Error"]["Message"]

                if (
                    error_code == "ThrottlingException"
                    and "Too many tokens" in error_message
                ):
                    logging.warning("Throttling detected: sleeping 5 seconds...")
                    failed_urls.append(url)
                    await asyncio.sleep(5)

                else:
                    logging.warning(f"❌ Error when judging {url}: {e}, continue")
                    failed_urls.append(url)

        final_malicious = page_malicious + screenshot_malicious
        logging.info(f"Token usage: {self.token_callback.usage_metadata}")

        return {
            "page_malicious": page_malicious,
            "screenshot_malicious": screenshot_malicious,
            "final_malicious": final_malicious,
            "failed_urls": failed_urls,
        }


class NoImgNodes:
    """A reactive URL classification node that operates without using images.

    This class uses a `CompiledStateGraph` agent to judge URLs based solely on textual content.
    It leverages a prompt (SYSTEM_NO_IMG) to determine whether a given URL is malicious or phishing,
    and aggregates the structured results, including failed attempts due to throttling.

    Args:
        react_agent (CompiledStateGraph): The state graph agent used to process input prompts.
        config (RunnableConfig): Runtime configuration passed to the agent.
        token_callback (Any): Callback for tracking token usage or metadata.
    """

    def __init__(
        self,
        react_agent: CompiledStateGraph,
        config: RunnableConfig,
        token_callback: UsageMetadataCallbackHandler,
    ):
        self.react_agent = react_agent
        self.config = config
        self.token_callback = token_callback

    async def react_judge_node(self, state: URLState) -> dict[str, Any]:
        """Processes a list of URLs using the reactive agent to determine if they are malicious.

        For each URL, a prompt is sent to the agent asking whether the site is malicious or phishing.
        The agent's response is parsed into structured verdicts, and any failures due to throttling are recorded.
        """
        verdicts, jsons, failed = [], [], []
        start = time.time()

        for i, url in enumerate(state["urls"]):
            agent_input = {
                "messages": [
                    SYSTEM_NO_IMG,
                    HumanMessage(
                        content=f"Judge if this URL: {url} is malicious or a phishing website."
                    ),
                ]
            }

            try:
                async for step in self.react_agent.astream(
                    agent_input, config=self.config, stream_mode="values"
                ):
                    last_msg = step["messages"][-1]

                final_msg = last_msg.content
                logging.info(final_msg)
                logging.info("===" * 50)

                verdicts.append({"url": url, "reason": final_msg})
                final_json = extract_and_fix(final_msg)
                for v in final_json[0]["verdicts"]:
                    jsons.append(v)

                if (i + 1) % 20 == 0:
                    elapsed = time.time() - start
                    avg = elapsed / (i + 1)
                    logging.info(f"Total elapsed: {elapsed:.2f}s")
                    logging.info(f"Avg time/URL: {avg:.2f}s")

            except botocore.exceptions.ClientError as e:
                code = e.response["Error"]["Code"]
                message = e.response["Error"]["Message"]
                if code == "ThrottlingException" and "Too many tokens" in message:
                    logging.warning("Throttling detected: sleeping 3 seconds…")
                    failed.append(url)
                    await asyncio.sleep(3)
                    continue
                logging.warning(f"❌ Error when judging {url}: {message}, continue")
                failed.append(url)
                continue

        return {
            "result": verdicts,
            "json_result": jsons,
            "failed_urls": failed,
        }

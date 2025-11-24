import argparse
import asyncio
import json
import logging
import time
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent

from agent_helpers import DeterministicNodes, NoImgNodes, ReactNodes
from callbacks import get_default_callbacks, get_token_usage_callbacks
from memory import AgenticMemorySystem, MemoryNodes
from state import ReactURLState, URLState
from tools import AgentTools
from utils import get_bedrock_client, get_llm

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("langchain_aws.llms.bedrock").setLevel(logging.WARNING)


def build_deterministic_agent(
    callbacks: Optional[list[BaseCallbackHandler]] = None,
) -> CompiledStateGraph:
    """Baseline: Judge whether a URL is malicious in a deterministic way."""
    # init llm with callbacks
    client = get_bedrock_client()
    token_callback = get_token_usage_callbacks()
    if callbacks is None:
        callbacks = []
    callbacks.append(token_callback)
    llm = get_llm(client, callbacks)

    # init tools with llm
    tools = AgentTools(llm)
    deterministic_nodes = DeterministicNodes(tools, token_callback)

    # assemble graph
    graph = StateGraph(URLState)
    graph.add_node("deterministic_judge", deterministic_nodes.process)
    graph.add_edge(START, "deterministic_judge")
    graph.add_edge("deterministic_judge", END)
    return graph.compile()


def build_noimg_agent(
    callbacks: Optional[list[BaseCallbackHandler]] = None,
) -> CompiledStateGraph:
    """Baseline: Judge whether a URL is malicious in using agent, without image-related tools."""
    # init llm with callbacks
    client = get_bedrock_client()
    llm = get_llm(client, callbacks)

    # init tools with llm
    tools = AgentTools(llm)

    # build react agent
    token_callback = get_token_usage_callbacks()
    if callbacks is None:
        callbacks = []
    callbacks.append(token_callback)
    config = RunnableConfig(callbacks=callbacks, recursion_limit=80)
    react_agent = create_react_agent(
        model=llm, tools=[tools.crawl, tools.extract_links]
    )

    nodes = NoImgNodes(
        react_agent=react_agent,
        config=config,
        token_callback=token_callback,
    )

    # assemble graph
    graph = StateGraph(URLState)
    graph.add_node("judge", nodes.react_judge_node)
    graph.add_edge(START, "judge")
    graph.add_edge("judge", END)
    return graph.compile()


def build_full_agent(
    callbacks: Optional[list[BaseCallbackHandler]] = None,
    use_memory: bool = True,
    memory_kwargs: Optional[dict[str, Any]] = None,
    args=None,
) -> CompiledStateGraph:
    """Builds a ReAct-based CompiledStateGraph agent with optional memory support."""
    # init llm with callbacks
    client = get_bedrock_client()
    llm = get_llm(client, callbacks)

    # init tools with llm
    tools = AgentTools(llm)
    tool_list = [
        tools.crawl,
        tools.extract_targets,
        tools.check_img,
        tools.check_screenshot,
        tools.serpapi_search,
    ]

    # Token usage callback
    token_callback = get_token_usage_callbacks()
    if callbacks is None:
        callbacks = []
    callbacks.append(token_callback)
    config = RunnableConfig(callbacks=callbacks, recursion_limit=80)

    # init memory
    if memory_kwargs is None:
        memory_kwargs = {}
    logging.info(f"Using memory: {use_memory}, Agent memory kwargs: {memory_kwargs}.")
    agent_memory = AgenticMemorySystem(llm, client, **memory_kwargs)
    memory_nodes = MemoryNodes(tools, agent_memory)
    react_nodes = ReactNodes(
        llm=llm,
        tools=tool_list,
        token_callback=token_callback,
        config=config,
        args=args,
    )

    # now wire up the StateGraph
    react_builder = StateGraph(ReactURLState, input=ReactURLState, config_schema=config)
    if use_memory:
        logging.info("Building graph with memory.")
        react_builder.add_node("prepare_memory", memory_nodes.prepare_memory)
        react_builder.add_node("store_memory", memory_nodes.store_memory)
        react_builder.add_node(react_nodes.call_model)
        react_builder.add_node("tools", ToolNode(tool_list))
        react_builder.add_edge("__start__", "prepare_memory")
        react_builder.add_edge("prepare_memory", "call_model")
        react_builder.add_conditional_edges(
            "call_model", react_nodes.route_model_output
        )
        react_builder.add_edge("tools", "call_model")
        react_builder.add_edge("store_memory", "__end__")
    else:
        logging.info("Building graph without memory.")
        react_builder.add_node("tools", ToolNode(tool_list))
        react_builder.add_node("store_memory", memory_nodes.store_memory)
        react_builder.add_node(react_nodes.call_model)
        react_builder.add_edge("__start__", "call_model")
        react_builder.add_conditional_edges(
            "call_model", react_nodes.route_model_output
        )
        react_builder.add_edge("tools", "call_model")

    # compile into the ReAct agent
    react_agent = react_builder.compile(name="ReAct Agent")
    react_nodes.react_agent = react_agent

    # assemble graph
    graph = StateGraph(URLState)
    graph.add_node("judge", react_nodes.react_judge_node)
    graph.add_edge(START, "judge")
    graph.add_edge("judge", END)
    return graph.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["determine", "noimg_agent", "full_agent"])
    parser.add_argument("--input", default="test.txt")
    parser.add_argument("--output", default="data.json")
    parser.add_argument(
        "--use-ai-overview",
        default=True,
        type=lambda x: (str(x).lower() in ("true", "1", "yes")),
        help="whether to use google ai overview in serpAPI (default: True)",
    )
    # memory related kwargs
    parser.add_argument(
        "--use-memory",
        default=True,
        type=lambda x: (str(x).lower() in ("true", "1", "yes")),
        help="whether to enable the memory-augmented reasoning (default: True)",
    )
    parser.add_argument(
        "-k", default=5, type=int, help="maximum similar memory returned"
    )
    parser.add_argument(
        "--threshold",
        default=0.60,
        type=float,
        help="similarity threshold for retrieving memories",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        urls = [u.strip().strip('",') for u in f if u.strip()]

    # pass callbacks specific to chosen agent
    callbacks = get_default_callbacks()
    tracker = callbacks[0]
    counter = callbacks[1]
    if args.agent == "determine":
        agent = build_deterministic_agent(callbacks=[tracker, counter])
    elif args.agent == "noimg_agent":
        agent = build_noimg_agent(callbacks=[tracker, counter])
    elif args.agent == "full_agent":
        agent = build_full_agent(
            callbacks=[tracker, counter],
            use_memory=args.use_memory,
            memory_kwargs={"k": args.k, "threshold": args.threshold},
            args=args,
        )
    else:
        raise NotImplementedError(f"{args.agent} is not supported.")

    start_time = time.time()
    output = asyncio.run(agent.ainvoke({"urls": urls}))
    total_time = time.time() - start_time
    avg_time = total_time / len(urls) if urls else 0
    logging.info(
        "Total running time: %.2fs.  Avg time: %.2fs per URL.", total_time, avg_time
    )

    result = output.get("result", [])
    json_result = output.get("json_result", [])
    failed_urls = output.get("failed_urls", [])

    logging.info("Tool usage: %s", dict(tracker.counts))
    logging.info("LLM calls: %d", counter.count)

    # Save output raw judge reason
    output_base = args.output.split(".")[0]
    with open(f"{output_base}_raw.json", "w") as f:
        json.dump(result, f, indent=2)

    # save urls
    if json_result:
        with open(args.output, "w") as f:
            json.dump(json_result, f, indent=2)
    if failed_urls:
        with open(f"{output_base}_failed_urls.txt", "w") as file:
            for line in failed_urls:
                file.write(line + "\n")

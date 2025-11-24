import logging
import uuid
from collections import Counter
from typing import Any, Optional

from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

from state import ReactURLState
from tools import AgentTools
from utils import extract_and_fix

logger = logging.getLogger(__name__)


class AgenticMemorySystem:
    def __init__(
        self,
        llm,
        client,
        evo_threshold: int = 100,
        k: int = 5,
        threshold: float = 0.60,
    ):
        embeddings = init_embeddings(
            "amazon.titan-embed-image-v1", provider="bedrock", client=client
        )
        self.memory_store = InMemoryStore(
            index={
                "embed": embeddings,
                "dims": 1024,
            }
        )
        # use its retriever for semantic search:
        self.k = k
        self.threshold = threshold

        # keep the LLM controller for summarization, etc.
        self.llm_controller = llm

        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

    async def summarize_keywords(self, text: str, screenshot_b64: str) -> list[str]:
        """
        Use the LLMController to extract a small set of salient keywords
        from the page text + screenshot.
        """
        system = {
            "role": "system",
            "content": (
                """Given the following text content and visual artifacts of a webpage, generate up to 10 keywords that best capture its content, using comma as the separator.
                Format your response as a JSON object with a 'keywords' field containing the selected text. 
                Example response format:{{"keywords": "keyword1, keyword2, keyword3"}}"""
            ),
        }
        user = {
            "role": "user",
            "content": (
                f"Page text:\n{text}\n\n"
                f"Screenshot base64 (for context, do NOT output raw data):\n"
                f"{screenshot_b64}…[truncated]"
            ),
        }
        # call the async LLMController
        resp = await self.llm_controller.ainvoke([system, user])
        # assume comma-separated
        return [kw.strip() for kw in resp.content.split(",") if kw.strip()]

    async def search_by_keywords(self, keywords: list[str]) -> Optional[str]:
        """
        Do a semantic-search over past memories using those keywords.
        Returns the top-k similar objects.
        """
        query = " ".join(keywords)
        namespace = ("agent_memory",)

        # perform a k-NN search in that namespace
        hits = self.memory_store.search(
            namespace,
            query=query,
            limit=self.k,
        )

        case_summaries = []
        for hit in hits:
            if hit.score < self.threshold:
                continue
            mem = hit.value
            url = mem["url"]
            kws = mem["keywords"]
            verdict = mem["verdict"]
            trace = mem["trace"]

            label = "Malicious" if verdict.get("malicious") else "Benign"
            conf = verdict.get("confidence", 0)
            reason = verdict.get("reason", "")

            case_summaries.append(
                "• **URL:** " + url + "\n"
                f"  - **Keywords:** {kws}\n"
                f"  - **Verdict:** {label} (confidence {conf}/5)\n"
                f"  - **Reason:** {reason}\n"
                f"  - **Tool calling trace:** {trace}\n"
            )
        if case_summaries:
            snippet = (
                "Previously, for similar URLs, we had these memories:\n\n"
                + "\n\n".join(case_summaries)
            )
        else:
            snippet = None

        return snippet

    async def search_by_keywords_w_majority(
        self, keywords: list[str]
    ) -> tuple[Optional[str], Optional[bool]]:
        """
        Semantic-search past memories, build a snippet, AND do a majority vote
        on malicious vs. benign. Returns (snippet, majority_malicious) where
        majority_malicious is True/False if there was a majority, or None.
        """
        query = " ".join(keywords)
        namespace = ("agent_memory",)
        hits = self.memory_store.search(
            namespace,
            query=query,
            limit=self.k,
        )

        verdicts = []
        case_summaries = []
        for hit in hits:
            if hit.score < self.threshold:
                continue
            mem = hit.value
            url = mem["url"]
            kws = mem["keywords"]
            verdict = mem["verdict"]
            trace = mem["trace"]

            is_mal = bool(verdict.get("malicious", False))
            verdicts.append(is_mal)

            label = "Malicious" if is_mal else "Benign"
            conf = verdict.get("confidence", 0)
            reason = verdict.get("reason", "")

            case_summaries.append(
                f"• **URL:** {url}\n"
                f"  - **Keywords:** {kws}\n"
                f"  - **Verdict:** {label} (confidence {conf}/5)\n"
                f"  - **Reason:** {reason}\n"
                f"  - **Too calling Trace:** {trace}\n"
            )

        if not case_summaries:
            return None, None

        # build the snippet
        snippet = (
            "Previously, for similar URLs, we had these memories:\n\n"
            + "\n".join(case_summaries)
        )

        # majority vote
        counts = Counter(verdicts)
        majority = False
        if (counts[True] > counts[False]) and len(case_summaries) >= self.k:
            majority = True

        return snippet, majority

    async def store_memory(
        self,
        keywords: list[str],
        trace: list[str],
        verdict: dict[str, Any],
        url: str,
    ) -> None:
        """
        Package url + keywords + tool-trace + final verdict
        and add it to the in-memory vector store.
        """
        content = {
            "url": url,
            "keywords": [", ".join(keywords)],
            "verdict": verdict,
            "trace": trace,
        }
        namespace = ("agent_memory",)
        memory_id = uuid.uuid4().hex
        self.memory_store.put(
            namespace=namespace,
            key=memory_id,
            value=content,
        )


class MemoryNodes:
    def __init__(self, tools: AgentTools, agent_memory: AgenticMemorySystem):
        """Nodes for integrating memory-based reasoning into the ReAct agent.

        This class is for preparing and storing memory associated with URL analysis,
        allowing the agent to reuse past reasoning results when similar content is encountered.

        Args:
            tools (AgentTools): A suite of tools used for crawling and analysis.
            agent_memory (AgenticMemorySystem): The memory system for storing and retrieving past interactions.
        """
        self.tools = tools
        self.agent_memory = agent_memory

    async def prepare_memory(self, state: ReactURLState) -> dict[str, Any]:
        """Prepares memory context before the main reasoning step.

        This function crawls the page, summarizes it into keywords,
        and attempts to retrieve relevant past memories using majority voting.
        """
        # crawl page & screenshot
        page = await self.tools.crawl.arun({"url": state.url, "screenshot": True})

        # summarize into keywords
        keywords = await self.agent_memory.summarize_keywords(
            page["text"], page["screenshot"]
        )
        logging.info(f"keywords for current url: {keywords} \n {state.url}")

        # retrieve past memory + majority vote
        retrieved_mem, majority = await self.agent_memory.search_by_keywords_w_majority(
            keywords
        )
        mem_case = "memory_reuse" if retrieved_mem else "full_reasoning"

        return {
            "memory_snippet": retrieved_mem,
            "keywords": keywords,
            "memory_majority": majority,
            "memory_case": mem_case,
        }

    async def store_memory(self, state: ReactURLState) -> dict[str, Any]:
        """Stores the current interaction into memory if the model is confident.

        Extracts the final AI response, parses the verdict, and saves it along with
        trace information and keywords for future reuse if confidence is high.

        """
        # grab the last AIMessage
        final_msg = state.messages[-1].content
        # extract the structured verdict
        final_json = extract_and_fix(final_msg)
        try:
            for v in final_json[0]["verdicts"]:
                if v["confidence"] > 4:
                    # Save memory when model is confident
                    logging.info(
                        f"Save memory {state.keywords}, {state.tool_sequence}, {v}"
                    )
                    await self.agent_memory.store_memory(
                        keywords=state.keywords,
                        trace=state.tool_sequence,
                        verdict=v,
                        url=state.url,
                    )
        except Exception as e:
            logging.info(f"Error {e}, model response: {final_msg}")

        # no outputs needed for downstream
        return {}

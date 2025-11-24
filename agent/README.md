# 🤖 Multi-Modal Phishing URL Detection Framework

This folder contains a modular framework for detecting phishing and impersonation URLs using LLM-based agents, deterministic workflows, and monolithc LLM prompting methods. It supports tool-based reasoning, recursive URL inspection, and multi-modal evidence including screenshots and images.

---

## 📁 Project Structure

```
src/
├── agent_helper.py           # Graph nodes for different agent designs
├── callbacks.py              # Custom callback handlers (e.g., token usage tracking)
├── graph.py                  # Entry point for compiling agent/baseline graphs
├── memory.py                 # Agentic memory system for agent
├── prompts.py                # System prompts and prompt templates
├── state.py                  # Shared TypedDict-based state definitions
├── tools.py                  # Tool wrappers (crawl, screenshot, image inspection, etc.)
├── utils.py                  # Helper utilities (LLM client initialization, retrievers, etc.)
└── baseline_monolithic.py    # Monolithc LLM baseline
```

## ⚙️ Prerequsites

1. Clone the repo

   ```bash
   git clone https://github.com/memophishagent/MemoPhishAgent.git
   cd agent
   ```
2. Install python packages inside Docker

   ```bash
   docker build -t my-agent .
   ```
3. Prepare input URLs

Place your URLs (one per line) into a `.txt` file. You may place this file in a new folder of your choice for better organization.


## 🧪 Baseline and Agent Designs

### 1. Monolithic LLM (Prompt-only)

- 🔹 **No tools**, no agent logic.
- 🔹 For each URL: fetch content → construct a single prompt to the LLM.
- 🔹 Prompt contains the URL, its page text, and optionally a screenshot.
- 🔹 LLM returns `{malicious, confidence, reason}` directly.
- ✅ Fast and simple, but limited in reasoning depth and modularity.

### 2. Deterministic Workflow

- 🔹 Fixed multi-step pipeline:

URL → crawl content → judge page → check screenshot → extract inside URLs/images → recurse

- 🔹 No agent reasoning, but **uses tools** in a hard-coded order.
- 🔹 Useful for baseline comparisons.
- ✅ Good balance between reasoning depth and cost, but not adaptive.

### 3. LLM-Driven Agent (Text-only)

- 🔹 Uses a ReAct-style LLM agent to decide **which tools to call and when**.
- 🔹 Supports recursion (e.g., crawl → extract → inspect children).
- 🔹 **No image-related tools**.
- ✅ Tests the value of dynamic tool reasoning with minimal modality input.

### 4. Full LLM Agent (Multi-modal)

- 🔹 ReAct-style agent with tool calling, recursion, and multi-modal support.
- 🔹 Tools include:
- `crawl_content`
- `check_screenshot`
- `extract_targets`
- `check_img`
- `serpapi_search`
- `google_ai_overview`
- 🔹 Agent autonomously chooses which tools to invoke based on page content and prompt instructions.
- ✅ Most powerful and accurate, at the cost of latency and token usage.

---

## 📊 Key Features

- ✅ **Multi-modal**: Incorporates screenshots and visual artifacts.
- ✅ **Agentic memory system**: Designs a noval agentic memory system to learn from history interactions.
- ✅ **Modular tools**: Each tool is a callable unit (can be swapped or extended).
- ✅ **Callback support**: Track token usage and reasoning steps.

---

## 🚀 Running Each Pipeline

Run the desired baseline or agent via:

```bash
docker run --rm -it -v "$(pwd)":/work -w /work my-agent python src/graph.py --mode full_agent --input urls.txt --output result.json
```
Available --mode options:
- deterministic – rule-based, no LLM involvement
- no_img – LLM agent without image-based reasoning
- full_agent – full LLM agent with multi-modal (text + image) analysis

The generated output includes:

• **Final URLs judge result**:
The json file of URLs judged as malicious or benign by the method (including its memory usage analysis).

• **Failed URLs:**
URLs that encountered errors during processing (e.g., network issues, throttling, or tool failures).

We can also customize the pipeline with the following optional flags:

| Argument              | Type               | Default | Description                                                                                   |
|-----------------------|--------------------|---------|-----------------------------------------------------------------------------------------------|
| `--use-ai-overview`   | bool  | `True`  | Whether to use Google’s AI Overview from SerpAPI to enrich search results. Set to  `False` to disable. |
| `--use-memory`        | bool  | `True`  | Enables memory-augmented reasoning, allowing the agent to recall and leverage similar past URLs. |
| `-k`                  | int                | `5`     | The maximum number of similar memory entries to retrieve when memory is enabled.             |
| `--threshold`         | float              | `0.60`  | Similarity threshold (0–1) for selecting relevant memories. Higher values yield stricter matching. |


## 📈 Metrics & Logging

✅ Tracks:

• Malicious URL detections

• Additional phishing URLs (e.g., child links)

• Recall, latency, and LLM calls per URL

✅ Callbacks:

• Token usage

• LLM calls for each tool

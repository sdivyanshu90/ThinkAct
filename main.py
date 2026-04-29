from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agent.agent_loop import ReActAgent
from agent.llm_client import OpenAICompatibleLLMClient, OpenAICompatibleProviderConfig, QueuedLLMClient, RuleBasedMockLLMClient
from agent.logging_utils import DebugTracer, configure_logger
from agent.models import AgentConfig, SearchDocument
from agent.parser import CommandParser
from agent.tool_registry import ToolRegistry
from agent.tools import DuckDuckGoSearchProvider, StaticSearchProvider, build_default_tools


DEFAULT_OPENROUTER_FREE_MODEL = "openai/gpt-oss-120b:free"


def load_dotenv(dotenv_path: Path | None = None) -> None:
    path = dotenv_path or Path(__file__).resolve().with_name(".env")
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()

        key, separator, value = line.partition("=")
        if not separator:
            continue

        normalized_key = key.strip()
        normalized_value = value.strip()
        if (
            len(normalized_value) >= 2
            and normalized_value[0] == normalized_value[-1]
            and normalized_value[0] in {"'", '"'}
        ):
            normalized_value = normalized_value[1:-1]

        os.environ.setdefault(normalized_key, normalized_value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the explicit ReAct agent loop.")
    parser.add_argument("--task", required=True, help="Task for the agent to solve.")
    parser.add_argument(
        "--provider",
        choices=("mock", "queued", "openrouter"),
        default="mock",
        help="Model provider implementation to use.",
    )
    parser.add_argument(
        "--model",
        help="Model ID for the selected provider. Required for real providers unless set via environment.",
    )
    parser.add_argument(
        "--response-script",
        type=Path,
        help="JSON file containing an array of scripted model responses for the queued provider.",
    )
    parser.add_argument(
        "--search-provider",
        choices=("duckduckgo", "static"),
        default="duckduckgo",
        help="Search backend to use when the agent calls the search tool.",
    )
    parser.add_argument(
        "--search-corpus",
        type=Path,
        help="Optional JSON file mapping queries to search result documents. Implies static search.",
    )
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum model turns before halting.")
    parser.add_argument("--debug", action="store_true", help="Print structured logs and a readable trace to stderr.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    try:
        logger = configure_logger(debug=args.debug)
        tracer = DebugTracer(enabled=args.debug)
        search_provider = load_search_provider(args.search_provider, args.search_corpus)
        client = build_client(args.provider, args.response_script, args.model)
        config = AgentConfig(max_steps=args.max_steps, debug=args.debug)
        parser = CommandParser(client=client, max_repair_attempts=config.max_parse_retries)

        registry = ToolRegistry()
        registry.register_many(build_default_tools(search_provider=search_provider))

        agent = ReActAgent(
            client=client,
            registry=registry,
            parser=parser,
            config=config,
            logger=logger,
            tracer=tracer,
        )
        result = agent.run(args.task)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if result.final_answer:
        print(result.final_answer)
        return 0

    print(f"Agent halted without a final answer: {result.reason}", file=sys.stderr)
    return 1


def build_client(provider_name: str, response_script: Path | None, model: str | None):
    if provider_name == "mock":
        return RuleBasedMockLLMClient()
    if provider_name == "queued":
        if response_script is None:
            raise ValueError("--response-script is required when --provider queued is selected")
        payload = json.loads(response_script.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError("response script must be a JSON array of strings")
        return QueuedLLMClient(responses=list(payload))

    resolved_model = (model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_FREE_MODEL).strip()

    api_key = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_COMPAT_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY or OPENAI_COMPAT_API_KEY must be set when --provider openrouter is selected")

    return OpenAICompatibleLLMClient(
        name="openrouter",
        config=OpenAICompatibleProviderConfig(
            api_key=api_key,
            model=resolved_model,
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            app_name="ThinkAct",
            app_url="https://example.local/thinkact",
            require_free_model_suffix=True,
        ),
    )


def load_search_provider(search_provider_name: str, search_corpus: Path | None):
    if search_corpus is None and search_provider_name == "duckduckgo":
        return DuckDuckGoSearchProvider()

    if search_corpus is None:
        return StaticSearchProvider(
            documents_by_query={
                "What is the capital of France?": [
                    SearchDocument(
                        title="France",
                        snippet="Paris is the capital of France.",
                        url="https://example.test/france",
                    )
                ],
                "Who created Python?": [
                    SearchDocument(
                        title="Python History",
                        snippet="Python was created by Guido van Rossum.",
                        url="https://example.test/python-history",
                    )
                ],
            }
        )

    payload = json.loads(search_corpus.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("search corpus must be a JSON object mapping query strings to document lists")

    documents_by_query: dict[str, list[SearchDocument]] = {}
    for query, docs in payload.items():
        if not isinstance(query, str) or not isinstance(docs, list):
            raise ValueError("invalid search corpus shape")
        documents_by_query[query] = [
            SearchDocument(title=doc["title"], snippet=doc["snippet"], url=doc["url"])
            for doc in docs
        ]
    return StaticSearchProvider(documents_by_query=documents_by_query)


if __name__ == "__main__":
    raise SystemExit(main())
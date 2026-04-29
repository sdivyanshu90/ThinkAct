# ThinkAct

ThinkAct is a minimal, production-style single-agent framework that implements an explicit ReAct loop with a strict JSON action protocol, pluggable tools, bounded memory, and deterministic state transitions.

The repository runs locally without any external API. The default CLI provider is a rule-based mock so the full loop can be exercised end-to-end out of the box. A real OpenRouter-backed provider (`:free` models only) and a live DuckDuckGo search backend are also included, using only the Python standard library.

## What the system does

- Accepts a user task.
- Builds a restrictive prompt from the task, available tools, and bounded recent memory.
- Calls a model provider.
- Parses the provider output into exactly one structured command.
- Executes a tool when the command is `tool_call`.
- Sanitizes the observation and feeds it back into the loop.
- Repeats until the model explicitly returns `final` or the runtime halts safely.

## How the loop works

The runtime is an explicit state machine. The states are:

- `INIT`
- `THINK`
- `ACT`
- `OBSERVE`
- `REFLECT`
- `FINAL`
- `ERROR`
- `HALT`

The normal happy path is:

1. `INIT`: store the user task in bounded memory.
2. `THINK`: build the prompt, call the model, and parse one JSON command.
3. `ACT`: dispatch the requested tool.
4. `OBSERVE`: sanitize the tool result before it can influence the next prompt.
5. `REFLECT`: append the step to memory and continue.
6. `FINAL`: store the final answer only when the model explicitly chooses to finish.
7. `HALT`: stop with a final answer or a safe termination reason.

Error paths transition through `ERROR` and then `HALT`.

## Output protocol

Every model turn must return exactly one JSON object and nothing else.

Tool call:

```json
{
  "type": "tool_call",
  "thought": "brief internal reasoning summary",
  "tool_name": "calculator",
  "tool_input": { "expression": "2 + 2" }
}
```

Final answer:

```json
{
  "type": "final",
  "thought": "brief internal reasoning summary",
  "final_answer": "The result is 4."
}
```

The parser rejects malformed output, extra keys, unsupported command types, invalid field shapes, and empty responses. It performs one repair pass through the model provider before halting with a structured parse error.

## File layout

```text
agent/
	__init__.py
	agent_loop.py
	llm_client.py
	logging_utils.py
	memory.py
	models.py
	parser.py
	prompt.py
	safety.py
	state.py
	tool_registry.py
	tools.py
tests/
	test_agent_loop.py
	test_llm_client.py
	test_parser.py
	test_safety.py
	test_tools.py
main.py
pyproject.toml
README.md
```

File-by-file summary:

- `agent/__init__.py`: package exports for core public types.
- `agent/agent_loop.py`: explicit ReAct state machine and orchestration handlers.
- `agent/llm_client.py`: model-provider abstraction; includes a queued scripted mock, a rule-based mock, and a real OpenAI-compatible HTTP client used for OpenRouter.
- `agent/logging_utils.py`: structured JSON logging and human-readable debug tracing.
- `agent/memory.py`: bounded short-term memory with trimming by message count and approximate size.
- `agent/models.py`: typed dataclasses for commands, memory records, configuration, tool results, and prompt context.
- `agent/parser.py`: strict JSON command parser with one repair attempt.
- `agent/prompt.py`: restrictive system prompt and deterministic prompt assembly.
- `agent/safety.py`: tool timeout wrapper, repeat detection, and prompt-injection redaction for observations.
- `agent/state.py`: `AgentState` enum and allowed state-transition table with enforcement.
- `agent/tool_registry.py`: pluggable tool registry and dispatch.
- `agent/tools.py`: tool interface plus `calculator`, `python_repl`, `search`, a real DuckDuckGo search provider, and the default tool builder.
- `tests/test_agent_loop.py`: end-to-end and loop-behavior tests.
- `tests/test_llm_client.py`: real-provider HTTP client tests with mocked transport.
- `tests/test_parser.py`: parser acceptance, rejection, and repair tests.
- `tests/test_safety.py`: prompt-injection redaction, repeat guard, and memory trimming tests.
- `tests/test_tools.py`: calculator, DuckDuckGo parsing, search, REPL, and registry tests.
- `main.py`: CLI entry point.
- `pyproject.toml`: project metadata and Python version requirement.

## How to run it

Arithmetic example:

```bash
python3 main.py --task "What is 17 * 19?"
```

Search example with the real DuckDuckGo backend:

```bash
python3 main.py --task "Who created Python?"
```

Offline/static search example:

```bash
python3 main.py --task "Who created Python?" --search-provider static
```

Debug mode prints both structured logs and a readable trace to stderr:

```bash
python3 main.py --task "What is 17 * 19?" --debug
```

Use a scripted queued model provider by passing a JSON array of model responses:

```json
[
  "{\"type\":\"tool_call\",\"thought\":\"use calculator\",\"tool_name\":\"calculator\",\"tool_input\":{\"expression\":\"2+2\"}}",
  "{\"type\":\"final\",\"thought\":\"done\",\"final_answer\":\"The result is 4.\"}"
]
```

```bash
python3 main.py --task "What is 2 + 2?" --provider queued --response-script responses.json
```

Use a real OpenRouter-backed free model:

```bash
export OPENROUTER_API_KEY="<your-openrouter-key>"
python3 main.py --provider openrouter --task "What is 17 * 19?"
```

The CLI defaults to the currently validated free model `openai/gpt-oss-120b:free`.

If you prefer, you can pass the model on the command line instead of `OPENROUTER_MODEL`:

```bash
export OPENROUTER_API_KEY="<your-openrouter-key>"
python3 main.py --provider openrouter --model "openai/gpt-oss-120b:free" --task "Summarize the capital of France."
```

## How to run tests

```bash
python3 -m unittest discover -s tests -v
```

## Safety boundaries

- Model output must be valid JSON and match one of two command schemas.
- The parser performs one repair pass and then halts safely.
- Unsupported tool names and invalid tool inputs are converted into structured tool failures.
- Tool execution uses a timeout wrapper.
- Repeated identical actions are detected and halt the run safely.
- Memory is explicitly bounded and trimmed.
- Tool outputs are treated as untrusted data and sanitized before reuse.
- Prompt-injection-like lines in tool outputs are redacted.
- The final answer is only emitted when the model explicitly returns a `final` command.

## Built-in tools

- `calculator`: AST-based arithmetic evaluator. No `eval`, no arbitrary code execution.
- `python_repl`: restricted interpreter for small local computations.
- `search`: provider-backed abstraction with a static mock implementation for local runs and tests.

Real search backend:

- The CLI defaults to `duckduckgo`, which fetches and parses DuckDuckGo Lite HTML using only the Python standard library.
- Pass `--search-provider static` or `--search-corpus path.json` when you want deterministic offline behavior.

## How to add a new tool

1. Implement a class in `agent/tools.py` (or a new tool module if you later split the file) that subclasses `Tool`.
2. Define `name`, `description`, and `input_schema`.
3. Implement `run(input_data: dict[str, Any]) -> ToolResult`.
4. Register the tool in `build_default_tools(...)` or in a custom registry assembly path.
5. Add unit tests and, if the tool changes loop behavior, at least one agent-loop test.

## Known limitations and tradeoffs

- The default CLI model is a deterministic mock provider, not a real LLM backend.
- The real provider path is wired for OpenRouter free models because free models are explicitly tagged with `:free` in OpenRouter's catalog, which makes the constraint enforceable programmatically.
- The CLI defaults to DuckDuckGo for live search; use `--search-provider static` or `--search-corpus` when deterministic results are needed.
- The `python_repl` is a restricted interpreter, not a hardened security sandbox. It blocks obvious filesystem, network, import, attribute, and process access patterns, but it should still be treated as a soft boundary.
- Tool timeouts are implemented with a thread-based wrapper for portability. This is enough for the included tools, but it does not forcibly terminate arbitrary native extensions.
- Memory trimming uses message count plus approximate character budget rather than tokenizer-specific token counting to keep dependencies minimal.

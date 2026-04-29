from __future__ import annotations

import ast
import html
from html.parser import HTMLParser
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from typing import Any, Mapping, Protocol, Sequence

from agent.models import SearchDocument, ToolResult, ToolSpec


class Tool(ABC):
    name: str
    description: str
    input_schema: Mapping[str, Any]

    @abstractmethod
    def run(self, input_data: dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )


class SearchProvider(Protocol):
    def search(self, query: str) -> Sequence[SearchDocument]:
        ...


@dataclass(slots=True)
class StaticSearchProvider:
    documents_by_query: dict[str, Sequence[SearchDocument]]

    def search(self, query: str) -> Sequence[SearchDocument]:
        return self.documents_by_query.get(query, ())


@dataclass(slots=True, frozen=True)
class DuckDuckGoSearchProvider:
    max_results: int = 5
    timeout_seconds: float = 10.0
    user_agent: str = "Mozilla/5.0 (ThinkAct)"
    base_url: str = "https://lite.duckduckgo.com/lite/"

    def search(self, query: str) -> Sequence[SearchDocument]:
        normalized_query = query.strip()
        if not normalized_query:
            return ()

        url = f"{self.base_url}?q={urllib_parse.quote(normalized_query)}"
        request = urllib_request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
            text = response.read().decode("utf-8", errors="replace")

        parser = _DuckDuckGoLiteParser(max_results=self.max_results)
        parser.feed(text)
        parser.close()
        return parser.results()


class _DuckDuckGoLiteParser(HTMLParser):
    def __init__(self, max_results: int) -> None:
        super().__init__(convert_charrefs=True)
        self._max_results = max_results
        self._results: list[SearchDocument] = []
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []
        self._current_href: str | None = None
        self._capturing_title = False
        self._capturing_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        if tag == "a" and attr_map.get("class") == "result-link":
            self._finalize_pending()
            self._capturing_title = True
            self._current_title = []
            self._current_snippet = []
            self._current_href = self._normalize_href(attr_map.get("href", ""))
            return

        if tag == "td" and attr_map.get("class") == "result-snippet":
            self._capturing_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capturing_title:
            self._capturing_title = False
            return

        if tag == "td" and self._capturing_snippet:
            self._capturing_snippet = False
            self._finalize_pending()

    def handle_data(self, data: str) -> None:
        if self._capturing_title:
            self._current_title.append(data)
        elif self._capturing_snippet:
            self._current_snippet.append(data)

    def close(self) -> None:
        super().close()
        self._finalize_pending()

    def results(self) -> list[SearchDocument]:
        return list(self._results[: self._max_results])

    def _finalize_pending(self) -> None:
        if len(self._results) >= self._max_results:
            return

        title = html.unescape("".join(self._current_title)).strip()
        href = (self._current_href or "").strip()
        snippet = self._normalize_whitespace(html.unescape("".join(self._current_snippet)))
        if title and href and not any(result.url == href for result in self._results):
            self._results.append(
                SearchDocument(
                    title=title,
                    snippet=snippet or "No snippet available.",
                    url=href,
                )
            )
        self._current_title = []
        self._current_snippet = []
        self._current_href = None

    def _normalize_href(self, href: str) -> str:
        if href.startswith("//"):
            href = f"https:{href}"
        parsed = urllib_parse.urlparse(href)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
            target = urllib_parse.parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                return urllib_parse.unquote(target)
        return href

    def _normalize_whitespace(self, text: str) -> str:
        return " ".join(text.split())


class CalculatorTool(Tool):
    name = "calculator"
    description = "Safely evaluates arithmetic expressions using numbers and arithmetic operators only."
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    _binary_nodes = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.FloorDiv: lambda left, right: left // right,
        ast.Mod: lambda left, right: left % right,
        ast.Pow: lambda left, right: left**right,
    }
    _unary_nodes = {
        ast.UAdd: lambda operand: +operand,
        ast.USub: lambda operand: -operand,
    }

    def run(self, input_data: dict[str, Any]) -> ToolResult:
        expression = input_data.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            return ToolResult(ok=False, content="", error="calculator requires a non-empty 'expression' string")

        try:
            parsed = ast.parse(expression, mode="eval")
            node_count = sum(1 for _ in ast.walk(parsed))
            if node_count > 64:
                raise ValueError("expression is too complex")
            value = self._evaluate(parsed.body)
        except ZeroDivisionError:
            return ToolResult(ok=False, content="", error="division by zero")
        except (SyntaxError, TypeError, ValueError) as exc:
            return ToolResult(ok=False, content="", error=f"invalid arithmetic expression: {exc}")

        return ToolResult(ok=True, content=str(value), metadata={"expression": expression})

    def _evaluate(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in self._binary_nodes:
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent is too large")
            return self._binary_nodes[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary_nodes:
            operand = self._evaluate(node.operand)
            return self._unary_nodes[type(node.op)](operand)

        raise ValueError(f"unsupported syntax: {type(node).__name__}")


class PythonReplTool(Tool):
    name = "python_repl"
    description = (
        "Executes a restricted subset of Python for local computation. "
        "It rejects imports, attribute access, and non-whitelisted builtins."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    _safe_builtins: dict[str, object] = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    _forbidden_nodes = (
        ast.Attribute,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.Nonlocal,
        ast.Raise,
        ast.Try,
        ast.While,
        ast.With,
        ast.AsyncWith,
        ast.Yield,
        ast.YieldFrom,
    )

    def run(self, input_data: dict[str, Any]) -> ToolResult:
        code = input_data.get("code")
        if not isinstance(code, str) or not code.strip():
            return ToolResult(ok=False, content="", error="python_repl requires a non-empty 'code' string")

        try:
            tree = ast.parse(code, mode="exec")
            self._validate_tree(tree)
            result = self._execute_tree(tree)
        except (SyntaxError, ValueError) as exc:
            return ToolResult(ok=False, content="", error=f"restricted python error: {exc}")
        except Exception as exc:  # pragma: no cover - defensive boundary
            return ToolResult(ok=False, content="", error=f"python execution failed: {exc}")

        return ToolResult(
            ok=True,
            content=result,
            metadata={"sandbox": "restricted-interpreter"},
        )

    def _validate_tree(self, tree: ast.Module) -> None:
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > 128:
            raise ValueError("code is too complex")

        for node in ast.walk(tree):
            if isinstance(node, self._forbidden_nodes):
                raise ValueError(f"node type is not allowed: {type(node).__name__}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in self._safe_builtins:
                    raise ValueError("only whitelisted builtin calls are allowed")
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise ValueError("private names are not allowed")
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                raise ValueError("comprehensions are not allowed")
            if isinstance(node, ast.For):
                raise ValueError("for loops are not allowed")

    def _execute_tree(self, tree: ast.Module) -> str:
        env: dict[str, object] = {}
        body = list(tree.body)
        trailing_expression: ast.Expr | None = None

        if body and isinstance(body[-1], ast.Expr):
            trailing_expression = body.pop()

        if body:
            executable = ast.Module(body=body, type_ignores=[])
            ast.fix_missing_locations(executable)
            exec(compile(executable, "<python_repl>", "exec"), {"__builtins__": self._safe_builtins}, env)

        if trailing_expression is not None:
            expression = ast.Expression(body=trailing_expression.value)
            ast.fix_missing_locations(expression)
            value = eval(  # noqa: S307 - input is AST-validated and builtins are restricted.
                compile(expression, "<python_repl>", "eval"),
                {"__builtins__": self._safe_builtins},
                env,
            )
            return repr(value)

        return repr(env)


@dataclass(slots=True)
class SearchTool(Tool):
    provider: SearchProvider
    name: str = "search"
    description: str = "Searches an external knowledge source through a provider abstraction."
    input_schema: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_schema",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    def run(self, input_data: dict[str, Any]) -> ToolResult:
        query = input_data.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(ok=False, content="", error="search requires a non-empty 'query' string")

        try:
            documents = self.provider.search(query.strip())
        except Exception as exc:  # pragma: no cover - defensive boundary
            return ToolResult(ok=False, content="", error=f"search provider failure: {exc}")

        if not documents:
            return ToolResult(ok=True, content="NO_RESULTS", metadata={"query": query, "hits": 0})

        lines = [
            f"{index}. {document.title}: {document.snippet} ({document.url})"
            for index, document in enumerate(documents, start=1)
        ]
        return ToolResult(
            ok=True,
            content="\n".join(lines),
            metadata={"query": query, "hits": len(documents)},
        )


def build_default_tools(search_provider: SearchProvider | None = None) -> list[Tool]:
    provider = search_provider or StaticSearchProvider(documents_by_query={})
    return [CalculatorTool(), PythonReplTool(), SearchTool(provider=provider)]
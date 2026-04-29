from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.models import SearchDocument, ToolResult
from agent.tool_registry import ToolRegistry
from agent.tools import CalculatorTool, DuckDuckGoSearchProvider, PythonReplTool, SearchTool, StaticSearchProvider, Tool


class _FakeHTMLResponse:
    def __init__(self, html_text: str) -> None:
        self._html = html_text.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._html


class EchoTool(Tool):
    name = "echo"
    description = "Echoes input"
    input_schema = {"type": "object"}

    def run(self, input_data: dict[str, object]) -> ToolResult:
        return ToolResult(ok=True, content=str(input_data["text"]))


class ToolTests(unittest.TestCase):
    def test_calculator_computes_basic_arithmetic(self) -> None:
        calculator = CalculatorTool()
        result = calculator.run({"expression": "17 * 19"})
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "323")

    def test_calculator_rejects_unsafe_expression(self) -> None:
        calculator = CalculatorTool()
        result = calculator.run({"expression": '__import__("os").system("echo nope")'})
        self.assertFalse(result.ok)
        self.assertIn("invalid arithmetic expression", result.error or "")

    def test_tool_registry_dispatches_correctly(self) -> None:
        registry = ToolRegistry()
        registry.register(EchoTool())
        result = registry.dispatch("echo", {"text": "hello"})
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "hello")

    def test_restricted_python_repl_blocks_imports(self) -> None:
        repl = PythonReplTool()
        result = repl.run({"code": "import os"})
        self.assertFalse(result.ok)
        self.assertIn("restricted python error", result.error or "")

    def test_search_tool_uses_provider(self) -> None:
        provider = StaticSearchProvider(
            {
                "Who created Python?": [
                    SearchDocument(
                        title="Python History",
                        snippet="Python was created by Guido van Rossum.",
                        url="https://example.test/python-history",
                    )
                ]
            }
        )
        tool = SearchTool(provider=provider)
        result = tool.run({"query": "Who created Python?"})
        self.assertTrue(result.ok)
        self.assertIn("Guido van Rossum", result.content)

    def test_duckduckgo_search_provider_parses_html_results(self) -> None:
        html_text = """
        <html>
          <body>
            <table>
              <tr>
                <td>1.</td>
                <td>
                  <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FGuido_van_Rossum&amp;rut=abc" class='result-link'>Guido van Rossum - Wikipedia</a>
                </td>
              </tr>
              <tr>
                <td class='result-snippet'>Guido van Rossum created Python.</td>
              </tr>
            </table>
          </body>
        </html>
        """

        provider = DuckDuckGoSearchProvider(max_results=3, timeout_seconds=1.0)
        with patch("agent.tools.urllib_request.urlopen", return_value=_FakeHTMLResponse(html_text)):
            results = provider.search("Who created Python?")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Guido van Rossum - Wikipedia")
        self.assertEqual(results[0].url, "https://en.wikipedia.org/wiki/Guido_van_Rossum")
        self.assertIn("created Python", results[0].snippet)
from __future__ import annotations

import unittest

from agent.llm_client import QueuedLLMClient
from agent.parser import CommandParser, ParseFailure


class CommandParserTests(unittest.TestCase):
    def test_accepts_valid_tool_call_json(self) -> None:
        parser = CommandParser()
        command = parser.parse(
            '{"type":"tool_call","thought":"use tool","tool_name":"calculator","tool_input":{"expression":"2+2"}}'
        )
        self.assertEqual(command.type, "tool_call")
        self.assertEqual(command.tool_name, "calculator")
        self.assertEqual(command.tool_input["expression"], "2+2")

    def test_accepts_valid_final_json(self) -> None:
        parser = CommandParser()
        command = parser.parse('{"type":"final","thought":"done","final_answer":"4"}')
        self.assertEqual(command.type, "final")
        self.assertEqual(command.final_answer, "4")

    def test_rejects_malformed_output(self) -> None:
        parser = CommandParser(max_repair_attempts=0)
        with self.assertRaises(ParseFailure):
            parser.parse("not-json")

    def test_attempts_single_repair_pass(self) -> None:
        client = QueuedLLMClient(
            responses=[
                '{"type":"final","thought":"repair","final_answer":"repaired"}',
            ]
        )
        parser = CommandParser(client=client, max_repair_attempts=1)
        command = parser.parse("not-json")
        self.assertEqual(command.type, "final")
        self.assertEqual(command.final_answer, "repaired")
        self.assertEqual(len(client.requests), 1)
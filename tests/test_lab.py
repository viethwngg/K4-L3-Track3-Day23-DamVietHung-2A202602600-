"""Offline regression tests. Run: python -m unittest discover -s tests -v."""
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage

import agents
import research
import tools
from check_citations import check
from finalize_citations import finalize


def source(n, family="web"):
    identifier = f"2501.{n:05d}"
    url = (f"https://arxiv.org/abs/{identifier}" if family == "arxiv" else
           f"https://huggingface.co/papers/{identifier}" if family.startswith("hf-") else f"https://example.org/{n}")
    return {"n": n, "id": identifier, "url": url, "title": f"Paper {n}", "date": "2025-01-01", "source": family}


def fixture():
    sources = [source(1, "arxiv"), source(2, "hf-search"), source(3)]
    report, sources, problems = finalize("# Survey\n\nEvidence [1][2][3].", sources)
    assert not problems
    return report.encode(), json.dumps(sources, indent=2).encode()


def task_messages():
    return [AIMessage(content="", tool_calls=[{"name": "task", "args": {"subagent_type": "researcher"},
                                               "id": str(n)} for n in range(3)],
                      usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30})]


class CitationTests(unittest.TestCase):
    def test_finalizer_output_and_adjacent_citations(self):
        report, sources = fixture()
        self.assertEqual(check(report.decode(), json.loads(sources)), [])

    def test_grouped_citations_and_ranges(self):
        sources = [source(n) for n in range(1, 4)]
        refs = "\n".join(f"[{s['n']}] Paper. {s['url']}" for s in sources)
        for citation in ("[1, 2, 3]", "[1-3]", "[1–3]", "[1, 2-3]"):
            with self.subTest(citation=citation):
                self.assertEqual(check(citation + "\n## References\n" + refs, sources), [])

    def test_code_and_markdown_links_do_not_count(self):
        for ignored in ("`[2]`", "``literal `[2]` ``", "```python\n[2]\n```", "~~~\n[2]\n~~~",
                        "    [2]\n", "[2](https://example.org/2)"):
            with self.subTest(ignored=ignored):
                report = "[1]\n" + ignored + "\n## References\n[1] Paper. https://example.org/1\n"
                self.assertEqual(check(report, [source(1)]), [])

    def test_missing_unknown_unused_and_duplicate_references(self):
        sources = [source(1), source(2)]
        report = "[1][9]\n## References\n[1] A. https://example.org/1\n[1] A. https://example.org/1"
        problems = "\n".join(check(report, sources))
        for expected in ("[9] cited but missing", "source [2] never cited", "missing reference line [2]", "duplicate reference line [1]"):
            self.assertIn(expected, problems)

    def test_one_url_per_reference_and_matching_url(self):
        for line, expected in (("A https://example.org/1 B https://example.org/2", "exactly one URL"),
                               ("A https://example.org/2", "does not match"), ("No URL", "exactly one URL")):
            self.assertIn(expected, "\n".join(check("[1]\n## References\n[1] " + line, [source(1)])))

    def test_bad_source_schemas_do_not_crash(self):
        for sources in (None, {}, [], [None], [{"n": True, "url": "ftp://bad"}],
                        [{"n": [], "url": []}], [source(1), source(1)]):
            with self.subTest(sources=sources):
                self.assertTrue(check("[1]\n## References\n[1] X https://example.org/1", sources))

    def test_missing_heading_and_reference_numbers_not_body_citations(self):
        self.assertIn("missing ## References", "\n".join(check("[1]", [source(1)])))
        self.assertIn("never cited", "\n".join(check("No citations\n## References\n[1] https://example.org/1", [source(1)])))


class RetryTests(unittest.TestCase):
    @patch("tools.random.uniform", return_value=0.25)
    @patch("tools.time.sleep")
    def test_backoff_jitter_cap_and_no_final_sleep(self, sleep, jitter):
        fn = Mock(side_effect=tools.RetryableError("temporary"))
        with self.assertRaises(tools.RetryableError):
            tools.with_retry(fn, attempts=4, base=1, cap=3)
        self.assertEqual(fn.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.25, 2.25, 3])

    @patch("tools.time.sleep")
    def test_retry_after_then_success(self, sleep):
        fn = Mock(side_effect=[tools.RetryableError("quota", 12), "ok"])
        self.assertEqual(tools.with_retry(fn, cap=8), "ok")
        sleep.assert_called_once_with(8)

    @patch("tools.time.sleep")
    def test_non_retryable_error_not_retried(self, sleep):
        fn = Mock(side_effect=ValueError("bug"))
        with self.assertRaises(ValueError):
            tools.with_retry(fn)
        self.assertEqual(fn.call_count, 1)
        sleep.assert_not_called()

    def test_http_statuses_transport_and_retry_after(self):
        for status in (429, 500, 502, 503, 504):
            response = httpx.Response(status, headers={"Retry-After": "7"}, request=httpx.Request("GET", tools.ARXIV_URL))
            with patch("tools.httpx.request", return_value=response), self.assertRaises(tools.RetryableError) as raised:
                tools._request("GET", tools.ARXIV_URL)
            self.assertEqual(raised.exception.retry_after, 7)
        with patch("tools.httpx.request", side_effect=httpx.ConnectError("offline")), self.assertRaises(tools.RetryableError):
            tools._request("GET", tools.ARXIV_URL)
        response = httpx.Response(400, request=httpx.Request("GET", tools.ARXIV_URL))
        with patch("tools.httpx.request", return_value=response), self.assertRaises(httpx.HTTPStatusError):
            tools._request("GET", tools.ARXIV_URL)
        self.assertIsNone(tools._retry_after("not-a-delay"))
        self.assertIsNone(tools._retry_after("nan"))
        self.assertGreater(tools._retry_after("Wed, 21 Oct 2099 07:28:00 GMT"), 0)


class SourceToolTests(unittest.TestCase):
    def test_empty_queries_do_not_use_network(self):
        with patch("tools._request") as request:
            self.assertEqual(tools.arxiv_search.invoke({"query": '"::: AND OR"'}), "NO RESULTS")
            self.assertEqual(tools.hf_search_papers.invoke({"query": " "}), "NO RESULTS")
            self.assertEqual(tools.web_search.invoke({"query": " "}), "NO RESULTS")
            request.assert_not_called()

    def test_arxiv_normalization_and_clean_query(self):
        xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.12345v3</id><title> A\n title </title><published>2025-01-02T00:00:00Z</published><summary> A\n summary </summary></entry></feed>'
        with patch("tools._arxiv_request", return_value=SimpleNamespace(text=xml)) as request:
            result = json.loads(tools.arxiv_search.invoke({"query": 'all:"world" AND all:model OR', "max_results": 200}))
        self.assertEqual(result[0]["url"], "https://arxiv.org/abs/2501.12345")
        self.assertEqual(result[0]["title"], "A title")
        self.assertEqual(request.call_args.args[0]["search_query"], "all:world AND all:model")
        self.assertEqual(request.call_args.args[0]["max_results"], 30)

    def test_arxiv_throttles_requests_with_virtual_clock(self):
        now, starts = [100.0], []
        def sleep(seconds):
            now[0] += seconds
        def request(*args, **kwargs):
            starts.append(now[0])
            return "ok"
        with patch("tools._arxiv_last_call", None), patch("tools.time.monotonic", side_effect=lambda: now[0]), patch("tools.time.sleep", side_effect=sleep), patch("tools._request", side_effect=request):
            for _ in range(3):
                tools._arxiv_request({})
        self.assertEqual(starts, [100, 103, 106])

    def test_hf_filter_sort_summary_and_missing_id(self):
        items = [{"paper": {"id": "a", "title": "World models", "summary": "Detailed summary", "ai_summary": "Short summary", "upvotes": 2}},
                 {"paper": {"id": "b", "title": "WORLD models", "upvotes": 8}}, {"paper": {"title": "No ID"}}]
        with patch("tools._request", return_value=SimpleNamespace(json=lambda: items)):
            daily = json.loads(tools.hf_daily_papers.invoke({"keyword": "world"}))
            search = json.loads(tools.hf_search_papers.invoke({"query": "world"}))
            self.assertEqual(tools.hf_daily_papers.invoke({"keyword": "missing"}), "NO RESULTS")
        self.assertEqual([record["id"] for record in daily], ["b", "a"])
        self.assertEqual(search[0]["summary"], "Short summary")
        self.assertEqual(search[0]["url"], "https://huggingface.co/papers/a")

    def test_every_tool_returns_error_instead_of_raising(self):
        cases = [(tools.arxiv_search, {"query": "world"}), (tools.hf_daily_papers, {}),
                 (tools.hf_search_papers, {"query": "world"}), (tools.web_search, {"query": "world"}),
                 (tools.web_fetch, {"url": "https://example.org"})]
        with patch("tools._request", side_effect=ValueError("bad response")):
            for fn, args in cases:
                with self.subTest(tool=fn.name):
                    self.assertTrue(fn.invoke(args).startswith("ERROR:"))

    def test_exa_sse_metadata_rate_limit_retry_and_authentication(self):
        failure = {"jsonrpc": "2.0", "id": 1, "result": {"_meta": {"rateLimitExceeded": True, "retryAfter": 0}, "content": [{"type": "text", "text": "quota"}]}}
        success = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "Evidence https://example.org"}]}}
        responses = [SimpleNamespace(text="data: " + json.dumps(payload) + "\n\n", headers={}) for payload in (failure, success)]
        with patch.dict(os.environ, {"EXA_API_KEY": "test-key-only"}), patch("tools._request", side_effect=responses) as request, patch("tools.time.sleep") as sleep:
            result = tools.web_search.invoke({"query": "survey"})
        self.assertEqual(result, "Evidence https://example.org")
        sleep.assert_called_once_with(0)
        self.assertEqual(request.call_count, 2)
        kwargs = request.call_args.kwargs
        self.assertEqual(kwargs["headers"]["x-api-key"], "test-key-only")
        self.assertTrue(kwargs["json"]["params"]["arguments"]["objective"])
        self.assertIn("text/event-stream", kwargs["headers"]["Accept"])

    def test_exa_json_rpc_error_and_secret_redaction(self):
        for payload in ({"error": {"code": -1, "message": "bad test-key-only"}},
                        {"result": {"isError": True, "content": [{"type": "text", "text": "bad test-key-only"}]}}):
            with patch.dict(os.environ, {"EXA_API_KEY": "test-key-only"}), patch("tools._request", return_value=SimpleNamespace(text=json.dumps(payload), headers={})):
                result = tools.web_fetch.invoke({"url": "https://example.org"})
                self.assertTrue(result.startswith("ERROR:"))
                self.assertNotIn("test-key-only", result)
        self.assertIn("[REDACTED]", tools._redact("https://mcp.exa.ai/mcp?exaApiKey=hidden"))

    def test_exa_fetch_array_and_length_limit(self):
        with patch("tools._exa_call", return_value="a" * 15000) as call:
            self.assertEqual(len(tools.web_fetch.invoke({"url": "https://example.org"})), 12000)
        self.assertEqual(call.call_args.args[1]["urls"], ["https://example.org"])
        self.assertTrue(tools.web_fetch.invoke({"url": "file:///secret"}).startswith("ERROR:"))

    def test_sse_keepalive_multiline_and_json(self):
        payload = {"result": {"content": []}}
        self.assertEqual(tools._rpc_payload(json.dumps(payload)), payload)
        self.assertEqual(tools._rpc_payload(': ping\n\nevent: message\ndata: {"result":\ndata: {"content": []}}\n\n'), payload)


class RunnerTests(unittest.TestCase):
    def test_slug_cannot_escape_output_directory(self):
        for topic, expected in (("../../x", "x"), ("", "topic"), ("...", "topic"),
                                ("Survey about World Model", "survey-about-world-model"), ("CON", "topic-con")):
            self.assertEqual(research.slugify(topic), expected)
        self.assertLessEqual(len(research.slugify("a" * 100)), 60)

    def test_summary_counts_only_lead_calls_and_usage(self):
        messages = task_messages() + [{"tool_calls": [{"name": "execute"}], "usage_metadata": {"input_tokens": 3, "output_tokens": 4}}]
        summary = research.summarize(messages, 1.26, "configured-model")
        self.assertEqual(summary["subagent_calls"], 3)
        self.assertEqual(summary["tokens"], {"input": 23, "output": 14})
        self.assertEqual(summary["elapsed_s"], 1.3)

    def test_invalid_downloads_write_nothing(self):
        report, sources = fixture()
        for bad_report, bad_sources in ((None, sources), (b"  ", sources), (report, b"oops"), (report, b"[]"), (b"[99]", sources)):
            with tempfile.TemporaryDirectory() as folder, patch("research.download", return_value={research.REPORT_PATH: bad_report, research.SOURCES_PATH: bad_sources}):
                destination = Path(folder) / "reports"
                with self.assertRaises(RuntimeError):
                    research.save_outputs(None, "topic", task_messages(), 1, "model", destination)
                self.assertFalse(destination.exists())

    def test_outputs_preserve_downloaded_bytes_and_true_metadata(self):
        report, sources = fixture()
        with tempfile.TemporaryDirectory() as folder, patch("research.download", return_value={research.REPORT_PATH: report, research.SOURCES_PATH: sources}):
            path = research.save_outputs(None, "../../topic", task_messages(), 1.26, "model", folder)
            self.assertEqual(path.read_bytes(), report)
            self.assertEqual(path.with_suffix(".sources.json").read_bytes(), sources)
            meta = json.loads(path.with_suffix(".meta.json").read_text())
            self.assertEqual(meta["source_families"], ["arxiv", "hf-search", "web"])
            self.assertEqual(meta["subagent_calls"], 3)

    def test_failed_bundle_publish_restores_previous_files(self):
        real_replace = os.replace
        calls = [0]
        def replace(*args):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError("simulated disk error")
            return real_replace(*args)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "one.md"
            path.write_bytes(b"old")
            with patch("research.os.replace", side_effect=replace), self.assertRaises(OSError):
                research._write_bundle(folder, {"one.md": b"new", "two.md": b"new"})
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual([p.name for p in Path(folder).iterdir()], ["one.md"])

    def test_false_source_labels_and_insufficient_coverage_rejected(self):
        sources = [source(1, "arxiv"), source(2, "hf-search"), source(3)]
        sources[0]["url"] = "https://example.org/1"
        with self.assertRaises(RuntimeError):
            research._validate_sources(sources)
        with self.assertRaises(RuntimeError):
            research._validate_sources([source(1)])

    def test_main_cleans_up_after_agent_failure(self):
        closed = []
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="")
        @contextmanager
        def sandbox():
            try:
                yield backend
            finally:
                closed.append(True)
        scripts = {research.VALIDATOR_PATH: research.VALIDATOR_SOURCE.read_bytes(), research.FINALIZER_PATH: research.FINALIZER_SOURCE.read_bytes()}
        graph = Mock()
        graph.invoke.side_effect = RuntimeError("model failed")
        with patch("research.make_model", return_value=Mock(model_name="fake")), patch("research.open_sandbox", sandbox), patch("research.upload"), patch("research.download", return_value=scripts), patch("research.build_lead_agent", return_value=graph), patch("research.save_outputs") as save, patch("sys.stderr"):
            self.assertEqual(research.main("topic"), 1)
        self.assertEqual(closed, [True])
        self.assertEqual(graph.invoke.call_args.kwargs["config"]["recursion_limit"], 1000)
        save.assert_not_called()

    def test_cli_empty_topic_and_all_stops_on_failure(self):
        with patch("sys.stderr"):
            self.assertEqual(research.main(" "), 2)
        with patch("research.main", side_effect=[0, 1]) as run:
            self.assertEqual(research.cli(["--all"]), 1)
        self.assertEqual(run.call_count, 2)


class AgentTests(unittest.TestCase):
    def test_real_deepagents_graph_builds_and_invokes_without_network(self):
        from deepagents.backends import StateBackend
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

        class FakeToolModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        graph = agents.build_lead_agent(StateBackend(), FakeToolModel(responses=[AIMessage(content="done")]))
        result = graph.invoke({"messages": [{"role": "user", "content": "No tools needed."}]},
                              config={"recursion_limit": 10})
        self.assertEqual(result["messages"][-1].content, "done")

    def test_every_subagent_and_lead_has_limits(self):
        with patch("agents.create_deep_agent", return_value="graph") as create:
            self.assertEqual(agents.build_lead_agent("sandbox", "model"), "graph")
        kwargs = create.call_args.kwargs
        for middleware in [kwargs["middleware"], *(s["middleware"] for s in kwargs["subagents"])]:
            self.assertTrue(any(isinstance(m, agents.ModelCallLimitMiddleware) for m in middleware))
            self.assertTrue(any(isinstance(m, agents.ToolCallLimitMiddleware) for m in middleware))
        self.assertTrue(any(isinstance(m, agents.TodoListMiddleware) for m in kwargs["middleware"]))
        self.assertEqual(agents.build_subagents()[1]["tools"], [tools.web_fetch])
        self.assertIsNot(agents.build_subagents()[0]["middleware"][0], agents.build_subagents()[0]["middleware"][0])


if __name__ == "__main__":
    unittest.main()

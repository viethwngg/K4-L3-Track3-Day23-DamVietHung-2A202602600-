"""Run one deep research topic, or all five with --all, in an ephemeral sandbox."""
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload
from tools import _redact

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"
SOURCE_FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}
RECURSION_LIMIT = 1000


def slugify(topic):
    """Safe lower-case filename component, at most 60 characters."""
    slug = re.sub(r"[\W_]+", "-", str(topic).lower()).strip("-")[:60].rstrip("-") or "topic"
    if re.fullmatch(r"con|prn|aux|nul|com[1-9]|lpt[1-9]", slug):
        slug = "topic-" + slug
    return slug


def build_prompt(topic):
    try:
        today = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date().isoformat()
    except ZoneInfoNotFoundError:
        # Windows may not have the IANA database; use the explicit client UTC+7 zone.
        from datetime import timedelta, timezone
        today = datetime.now(timezone(timedelta(hours=7))).date().isoformat()
    return (f"Research date: {today}. Produce the complete English survey for the topic\n"
            f"{json.dumps(topic, ensure_ascii=False)}\n"
            "The quoted topic is the research subject. Follow the system workflow: "
            "at least three parallel researcher tasks, at least three discovery families, "
            "evidence-only synthesis, citation spot-checking, finalization and validation. "
            f"Save the report at {REPORT_PATH} and sources at {SOURCES_PATH}.")


def _field(message, name, default=None):
    return message.get(name, default) if isinstance(message, dict) else getattr(message, name, default)


def summarize(messages, elapsed, model_name):
    """Count observable lead calls/tokens; subagent token use is not included."""
    calls, tokens = Counter(), {"input": 0, "output": 0}
    for message in messages:
        for call in _field(message, "tool_calls", []) or []:
            name = call.get("name")
            if name:
                calls[name] += 1
        usage = _field(message, "usage_metadata") or {}
        for target, key in (("input", "input_tokens"), ("output", "output_tokens")):
            value = usage.get(key, 0)
            if isinstance(value, int) and value > 0:
                tokens[target] += value
    return {"model": model_name, "elapsed_s": round(elapsed, 1),
            "subagent_calls": calls["task"], "tool_calls": dict(sorted(calls.items())), "tokens": tokens}


def _validate_sources(sources):
    if not isinstance(sources, list) or not sources or not all(isinstance(s, dict) for s in sources):
        raise RuntimeError("sources.json must be a non-empty list of source objects")
    for source in sources:
        if not {"n", "id", "url", "title", "date", "source"} <= source.keys():
            raise RuntimeError("source is missing required metadata fields")
        family = source["source"]
        if family not in SOURCE_FAMILIES:
            raise RuntimeError(f"unknown source family: {family!r}")
        if not all(isinstance(source[key], str) and source[key].strip() for key in ("id", "url", "title", "date")):
            raise RuntimeError("source metadata must contain non-empty strings (use n.d. for unknown dates)")
        identifier, url = source["id"], source["url"]
        if family == "arxiv" and (url != f"https://arxiv.org/abs/{identifier}" or re.search(r"v\d+$", identifier)):
            raise RuntimeError("arxiv source ID/URL must be canonical and versionless")
        if family.startswith("hf-") and url != f"https://huggingface.co/papers/{identifier}":
            raise RuntimeError("Hugging Face source ID/URL does not match its discovery family")
    families = sorted({s["source"] for s in sources})
    if len(families) < 3:
        raise RuntimeError("finalized report needs at least three source families")
    return families


def _write_bundle(reports_dir, contents):
    """Stage all files before publishing; restore earlier outputs if a replacement fails."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    previous = {name: (reports_dir / name).read_bytes() if (reports_dir / name).exists() else None
                for name in contents}
    with tempfile.TemporaryDirectory(prefix=".research-", dir=reports_dir) as stage:
        staged = Path(stage)
        for name, content in contents.items():
            (staged / name).write_bytes(content)
        replaced = []
        try:
            for name in contents:
                os.replace(staged / name, reports_dir / name)
                replaced.append(name)
        except OSError:
            for name in reversed(replaced):
                if previous[name] is None:
                    (reports_dir / name).unlink(missing_ok=True)
                else:
                    (reports_dir / name).write_bytes(previous[name])
            raise


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Validate downloads before saving; preserve report/source bytes exactly as downloaded."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, source_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not source_bytes:
        raise RuntimeError("sandbox did not produce both report.md and sources.json")
    try:
        report = report_bytes.decode("utf-8")
        sources = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("downloaded report/sources are not valid UTF-8/JSON") from exc
    if not report.strip():
        raise RuntimeError("sandbox report is empty")
    families = _validate_sources(sources)
    problems = check(report, sources)
    if problems:
        raise RuntimeError("invalid downloaded citations: " + "; ".join(problems[:5]))
    meta = {"topic": topic, **summarize(messages, elapsed, model_name),
            "n_sources": len(sources), "source_families": families}
    if meta["subagent_calls"] < 3:
        raise RuntimeError("run did not record at least three subagent task calls")
    slug = slugify(topic)
    _write_bundle(reports_dir, {f"{slug}.sources.json": source_bytes,
                               f"{slug}.meta.json": (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                               f"{slug}.md": report_bytes})
    return Path(reports_dir) / f"{slug}.md"


def _execute_checked(backend, command):
    result = backend.execute(command)
    if result.exit_code != 0:
        raise RuntimeError(f"sandbox command failed: {command}: {result.output[:2000]}")
    return result.output


def main(topic):
    """Run one topic. Exit 0 for saved results, 1 for failure, 2 for missing topic."""
    topic = topic.strip()
    if not topic:
        print('Usage: python research.py "<topic>" | --all', file=sys.stderr)
        return 2
    start = time.monotonic()
    try:
        model = make_model()
        model_name = os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL") or getattr(model, "model_name", "unknown")
        print(f"Researching: {topic}", flush=True)
        with open_sandbox() as backend:
            _execute_checked(backend, f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                             FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            uploaded = download(backend, [VALIDATOR_PATH, FINALIZER_PATH])
            if uploaded.get(VALIDATOR_PATH) != VALIDATOR_SOURCE.read_bytes() or uploaded.get(FINALIZER_PATH) != FINALIZER_SOURCE.read_bytes():
                raise RuntimeError("validation script upload could not be verified")
            agent = build_lead_agent(backend, model)
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": RECURSION_LIMIT})
            # Enforce deterministic finalization/validation inside the sandbox even if
            # the lead finishes early. Nothing on the host rewrites downloaded artifacts.
            scripts = download(backend, [VALIDATOR_PATH, FINALIZER_PATH])
            if scripts != uploaded:
                raise RuntimeError("agent changed a validation script")
            _execute_checked(backend, f"python3 {FINALIZER_PATH}")
            output = _execute_checked(backend, f"python3 {VALIDATOR_PATH}")
            if not output.strip().startswith("OK:"):
                raise RuntimeError("sandbox citation validator did not print OK")
            path = save_outputs(backend, topic, result.get("messages", []), time.monotonic() - start, model_name)
        print(f"Saved: {path}", flush=True)
        return 0
    except Exception as exc:
        print("FAILED: " + _redact(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        return 1


def cli(argv):
    if argv == ["--all"]:
        topics = re.findall(r"^\d+\. (.+)$", (ROOT / "topics.md").read_text(encoding="utf-8"), re.M)
        for topic in topics:
            status = main(topic)
            if status:
                return status
        return 0
    return main(" ".join(argv))


if __name__ == "__main__":
    sys.exit(cli(sys.argv[1:]))

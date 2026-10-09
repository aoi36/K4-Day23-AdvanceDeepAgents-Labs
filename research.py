"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Safe file name from topic."""
    slug = re.sub(r"[^\w]+", "-", topic.lower()).strip("-")
    return (slug or "topic")[:60]


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return f"Research and write the final English survey report for this topic: {topic}. Follow every workflow step in your system prompt, including delegation, notes, sources.json, finalization, validation, and citation spot-checking."


def summarize(messages, elapsed, model_name):
    """Return summary metadata dict from lead messages."""
    tool_counts = Counter()
    tokens = {"input": 0, "output": 0}
    subagent_calls = 0

    for msg in messages:
        # tool calls
        calls = getattr(msg, "tool_calls", None)
        if calls:
            for call in calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name:
                    tool_counts[name] += 1
                    if name == "task":
                        subagent_calls += 1

        # token usage
        meta = getattr(msg, "usage_metadata", None)
        if isinstance(meta, dict):
            tokens["input"] += meta.get("input_tokens", 0) or 0
            tokens["output"] += meta.get("output_tokens", 0) or 0

    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_counts),
        "tokens": tokens,
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download report from sandbox and write three files into reports_dir."""
    responses = backend.download_files([REPORT_PATH, SOURCES_PATH])
    errors = [f"{response.path}: {response.error}" for response in responses if response.error]
    if errors:
        raise RuntimeError("sandbox download failed: " + "; ".join(errors))
    files = {response.path: response.content for response in responses}

    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("report.md is missing or empty")
    if not sources_bytes:
        raise RuntimeError("sources.json is missing")
    try:
        sources = json.loads(sources_bytes)
        if not isinstance(sources, list):
            raise ValueError("not a list")
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"sources.json is invalid: {exc}")

    slug = slugify(topic)
    reports_dir.mkdir(parents=True, exist_ok=True)

    meta = summarize(messages, elapsed, model_name)
    meta["topic"] = topic
    meta["n_sources"] = len(sources)
    meta["source_families"] = sorted({s.get("source") for s in sources if s.get("source")})

    report_path = reports_dir / f"{slug}.md"
    report_path.write_bytes(report_bytes)
    (reports_dir / f"{slug}.sources.json").write_bytes(sources_bytes)
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return report_path


def complete_report(agent, backend, result):
    """Allow one continuation for unfinished work, then require sandbox validation."""
    messages = result.get("messages", [])
    probe = backend.execute(f"test -s {REPORT_PATH} && test -s {SOURCES_PATH}")
    if probe.exit_code != 0:
        last_reply = str(getattr(messages[-1], "content", ""))[-2000:] if messages else "No agent messages returned."
        calls = summarize(messages, 0, "")["tool_calls"]
        metadata = getattr(messages[-1], "response_metadata", {}) if messages else {}
        finish_reason = metadata.get("finish_reason", "unknown")
        malformed = finish_reason == "MALFORMED_FUNCTION_CALL"
        if not calls and not malformed:
            raise RuntimeError(f"agent made no tool calls (finish_reason={finish_reason}); check model/endpoint tool-calling support. Last reply: {last_reply}")
        if "limit" in last_reply.lower():
            raise RuntimeError(f"agent stopped at a limit before producing output. Tool calls: {calls}. Last reply: {last_reply}")
        # ponytail: one continuation only; use checkpoints if resumable multi-run research is needed.
        result = agent.invoke(
            {"messages": [*messages, {"role": "user", "content": (
                "The previous response may have contained a malformed function call. Call only registered tools using their exact JSON schemas; "
                "do not emit Python code or invented function names as tool calls. If you have not planned yet, call write_todos first. "
                f"Your run ended without the required output files. Continue using any existing notes in {WORKDIR}/research/notes; "
                f"do not restart the research. Write {SOURCES_PATH} and {REPORT_PATH} using only retrieved evidence. "
                f"Run python3 {FINALIZER_PATH} and python3 {VALIDATOR_PATH}. "
                "If blocked, explain the precise tool error; do not claim success without the files."
            )}]},
            config={"recursion_limit": 1000},
        )
        messages = result.get("messages", [])
    validation = backend.execute(f"python3 {VALIDATOR_PATH}")
    if validation.exit_code != 0:
        last_reply = str(getattr(messages[-1], "content", ""))[-2000:] if messages else "No agent messages returned."
        metadata = getattr(messages[-1], "response_metadata", {}) if messages else {}
        raise RuntimeError(f"sandbox report validation failed: {validation.output}\n"
                           f"Model finish_reason: {metadata.get('finish_reason', 'unknown')}\nLast agent reply: {last_reply}")
    return messages


def configure_local_model(model):
    """Allow slower CPU inference without changing hosted-provider behavior."""
    base_url = os.getenv("LAB_BASE_URL") or os.getenv("OPENAI_ENDPOINT") or ""
    if urlsplit(base_url).hostname in {"localhost", "127.0.0.1", "::1"}:
        # ponytail: local inference gets 10 minutes; use GPU acceleration if this is still too slow.
        client = model.root_client.with_options(timeout=600, max_retries=0)
        async_client = model.root_async_client.with_options(timeout=600, max_retries=0)
        model = model.model_copy(update={
            "request_timeout": 600, "max_retries": 0,
            "root_client": client, "client": client.chat.completions,
            "root_async_client": async_client, "async_client": async_client.chat.completions,
        })
    return model


def main(topic):
    """Return process exit code."""
    topic = topic.strip()
    if not topic:
        print("Usage: python research.py <topic>", file=sys.stderr)
        return 2

    start = time.monotonic()

    try:
        model = configure_local_model(make_model())
        with open_sandbox() as backend:
            setup = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if setup.exit_code != 0:
                raise RuntimeError(f"sandbox setup failed: {setup.output}")
            upload(backend, {
                VALIDATOR_PATH: (ROOT / "check_citations.py").read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000},
            )
            messages = complete_report(agent, backend, result)
            elapsed = time.monotonic() - start
            try:
                report_path = save_outputs(backend, topic, messages, elapsed, model.model_name if hasattr(model, "model_name") else str(model), )
            except RuntimeError as exc:
                print(f"FAILED: {exc}", file=sys.stderr)
                return 1
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))

"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not sources:
        return ["no sources in sources.json"]

    source_map = {}
    seen_urls = {}
    for entry in sources:
        n = entry.get("n")
        if not isinstance(n, int):
            problems.append(f"source entry has non-integer n: {n!r}")
            continue
        url = entry.get("url", "")
        if not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] has invalid url: {url!r}")
        if url in seen_urls:
            problems.append(f"source [{n}] has duplicate url (same as [{seen_urls[url]}]): {url}")
        else:
            seen_urls[url] = n
        source_map[n] = entry

    ref_split = re.split(r"(?m)^##[ \t]+References[ \t]*$", report_text)
    if len(ref_split) < 2:
        problems.append("missing '## References' heading")
        body = report_text
        ref_section = ""
    else:
        body = ref_split[0]
        ref_section = ref_split[-1]

    # parse body citations: handle [n], [n, m], [n-m] but not code/links
    code_re = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
    segments = code_re.split(body)
    cited = set()
    group_re = re.compile(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\](?!\()")
    for i, seg in enumerate(segments):
        if i % 2:  # code segment
            continue
        for m in group_re.finditer(seg):
            for part in re.split(r"\s*,\s*", m.group(1)):
                span = re.fullmatch(r"(\d+)\s*[\u2013-]\s*(\d+)", part)
                if span:
                    a, b = int(span.group(1)), int(span.group(2))
                    if 0 <= b - a <= 200:
                        cited.update(range(a, b + 1))
                    else:
                        cited.update([a, b])
                else:
                    cited.add(int(part))

    for n in sorted(cited):
        if n not in source_map:
            problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(source_map):
        if n not in cited:
            problems.append(f"source [{n}] never cited")

    # check reference lines
    ref_line_re = re.compile(r"^\s*\[(\d+)\]", re.MULTILINE)
    url_re = re.compile(r"https?://\S+")
    ref_numbers = []
    ref_seen = {}
    for line in ref_section.splitlines():
        m = ref_line_re.match(line)
        if not m:
            continue
        rn = int(m.group(1))
        ref_numbers.append(rn)
        if rn in ref_seen:
            problems.append(f"reference [{rn}] appears more than once")
        ref_seen[rn] = True
        if rn not in source_map:
            problems.append(f"reference [{rn}] is not a source")
        urls_in_line = url_re.findall(line)
        # strip trailing punctuation from URLs
        urls_in_line = [u.rstrip(")>,;.") for u in urls_in_line]
        if len(urls_in_line) == 0:
            problems.append(f"reference [{rn}] has no URL")
        elif len(urls_in_line) > 1:
            problems.append(f"reference [{rn}] has multiple URLs (must be exactly one)")
        elif rn in source_map and urls_in_line[0] != source_map[rn].get("url", ""):
            problems.append(f"reference [{rn}] URL mismatch: got {urls_in_line[0]}, expected {source_map[rn]['url']}")

    if ref_section:
        for n in sorted(source_map):
            if n not in ref_seen:
                problems.append(f"source [{n}] missing from References section")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
